#!/usr/bin/python3
from quantel.ints.pyscf_integrals import PySCFIntegrals, PySCFMolecule
from quantel.opt.hybrid_ef import HybridEF
import numpy as np
import subprocess
import glob
import bisect 
from .pesUtils import *
import os, sys, uuid 

class PESWalker():
    
    def __init__(self, sol, geometries, proQueue, config, parent=None, tid=None):
        self.proQueue = proQueue 
        self.parent = parent
        self.sol = sol
        self.geoms = list(geometries)
        self.prop_geoms = []
        self.HessInd = None
        # Task id used by the parent to tell a finished task from a dead one
        self.tid = tid
        self.merged = None

        # Wavefuntion
        self.method = config["wfn"]
        if config["wfn"]=="uhf":
            from quantel.wfn.uhf import UHF
            self.WFN = UHF
        elif config["wfn"]=="csf": 
            from quantel.wfn.csf import CSF
            self.WFN = CSF 
            self.spin_coupling = ""  
        else: 
            raise RuntimeError(" Wavefunction method not recognised") 

        if config["ProOpt"]=="eigenvector_following": 
            from quantel.opt.eigenvector_following import EigenFollow 
            self.PropOPT = EigenFollow
        elif config["ProOpt"]=="lsr1": 
            from quantel.opt.lsr1 import SR1
            self.PropOPT = SR1
        else: 
            raise RuntimeError(" ProOpt method not recognised") 
         
        self.basis = config["mol"]["basis"]
        self.unit = config["mol"]["units"]
        self.spin = config["mol"]["spin"]
        self.charge = config["mol"]["charge"] 
       
        self.CoalSigThresh = config["thresh"]["CoalSigThresh"]
        self.cuspHessThresh = config["thresh"]["cuspHessThresh"] 
        # Underwhich we identify these as coalescing partners  
        self.CoalOverlapThresh = config["thresh"]["CoalOverlapThresh"] 
      
        # Hessian information
        self.nstore_hess = config["jobcontrol"]["nstore_hess"]
        self.nhess_elements = 10 
        self.hess_eigvals = []
        self.hess_eigvecs = []
        # Child process selection  
        self.setOffDiscont = config["jobcontrol"]["setOffDiscount"] 
        self.setOffFalseCoal = config["jobcontrol"]["setOffFalseCoal"]
        self.FalseCoalEWind = 1.0 

        self.restrict_within_interval_FalseCoal = config["jobcontrol"]["restrict_within_interval_FalseCoal"] 

        # Geometry information
        self.geomMax = config["geoms"]["geomMax"] 
        self.geomMin = config["geoms"]["geomMin"]
        self.includeSign = config["geoms"]["includeSign"]
        self.leading_zeros = config["geoms"]["leading_zeros"]
        self.geomDecimals = config["geoms"]["geomDecimals"]
        self.fineGrain = 10**(-self.geomDecimals) 
        self.coarseGrain = config["geoms"]["coarseGrain"]
        self.makeGeomscript = config["geoms"]["makeGeomscript"] 
        self.new_geoms = []

        # To assign the hessian index
        try: 
            self.exactHess = config["exactHess"]
        except: 
            self.exactHess = True 

        self.hessThresh = 1e-16
        # Deduplication thresholds 
        self.dedupEnergyThresh = 1e-8          
        self.dedupOverlapThresh = 1e-5    
        # Threshold for continuity backtracking to current solution 
        self.contEnergyThresh  = 1e-4     
        self.contOverlapThresh = 1e-4  
        # Soft modes 
        self.modes = []

        # Energy threshold
        try:  
            self.energy_thresh = config["branches"]["energy_thresh"]    
        except: 
            self.energy_thresh = None
 
    def _redirect_to_logfile(self, tag):
        """Point this worker's stdout and stderr at logs/<tag>.log.

           The swap is done with dup2 on the file descriptors, not by rebinding
           sys.stdout: quantel and PySCF write to fd 1 from compiled code and
           would otherwise keep spraying the shared terminal while the Python
           prints went to the file.  maxtasksperchild=1 means this process runs
           exactly one task, so it is a once-per-process swap with nothing to
           undo afterwards."""
        path = f"logs/{tag}.log"
        sys.stdout.flush()
        sys.stderr.flush()
        # O_APPEND is what lets fd 1 and fd 2 share one file safely - the
        # kernel seeks to EOF before every write, so stderr cannot overwrite
        # stdout from offset zero.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        # Rebuild Python's wrappers line-buffered.  fd 1 is a regular file now,
        # so Python would otherwise pick 8k block buffering and `tail -f` would
        # sit blank for minutes.  closefd=False: fd 1 must outlive these objects.
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
        return path

    def _check_in(self, what):
        """Must be the first thing a task does: claim a private log file, then
           tell the parent which pid is running this task so that a worker that
           is killed outright is distinguishable from one still working."""
        path = self._redirect_to_logfile(f"{self.tid}_{what}")
        print(f"=== {self.sol} {what} pid={os.getpid()} tid={self.tid} ===",
              flush=True)
        if self.tid is not None:
            self.proQueue.put(("alive", {"tid": self.tid, "pid": os.getpid(),
                                    "log": path}))
        return

    def get_wfn(self, geom, temp=False):
        """ Returns a newly initialised wavefunction object at specified geometry """
        if temp:
            mol = PySCFMolecule(f"temp/{geom}.xyz", self.basis, self.unit, spin=self.spin, charge=self.charge)
        else:
            mol = PySCFMolecule(geom+"/geom.xyz", self.basis, self.unit, spin=self.spin, charge=self.charge)
        ints = PySCFIntegrals(mol)
      
        # We want to be able to get the wave function with the correct arguments, 
        # for instance the correct spin coupling pattern

      
        if self.method=="csf":
            return self.WFN(ints, self.spin_coupling)
        else: 
            return self.WFN(ints) 
        
    #----Geometry scripts (read, name, create, store) ----------------------------------------------
    # Change the geom name formatting here!
    @staticmethod
    def geom_value(geom):
        # geom name is "geom_xxxx"
        return float(geom[5:])


    def make_geom_name(self, value):
        """Inverse of geom_value, matching the zero-padded on-disk naming
           """
        # 
        if self.includeSign: 
            return f"geom_{value:+0{self.geomDecimals + self.leading_zeros + 1}.{self.geomDecimals}f}"
        else:
            return f"geom_{value:0{self.geomDecimals + self.leading_zeros + 1}.{self.geomDecimals}f}"


    def _grid_between(self, from_geom, to_geom, step):
        """
            Interior grid values strictly between the two geometries, in walk
            order, clipped to [geomMin, geomMax].
        """
        if step < 10 **(-self.geomDecimals):
            raise ValueError(
                f"grid step {step} is finer than the {self.geomDecimals}-decimal "
                f"geometry naming; distinct points would collide on one directory")
        a, b = self.geom_value(from_geom), self.geom_value(to_geom)
        nstep = int(round(abs(b - a) / step))
        sign = 1.0 if a < b else -1.0
        grid = np.round(a + sign * step * np.arange(1, nstep), self.geomDecimals) + 0.0
        grid = [ x for x in grid if  self.geomMax >= x >= self.geomMin ]
        return grid


    def _generate_geometry(self, geomval, path):
        """Run make_geometry.py for geomval and leave the result at `path`.
           Return False if the geometry could not be generated."""
        # pid in the temp name: two processes must never share one staging file.
        tmp = f"{path}.tmp{os.getpid()}"

        def _discard():
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

        try:
            with open(tmp, "w") as fh:
                subprocess.run([sys.executable, self.makeGeomscript, str(geomval)],
                               stdout=fh, check=True)
                fh.flush()
                os.fsync(fh.fileno())
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"geometry generation failed for {path}: {exc!r}", flush=True)
            _discard()
            return False
        if os.path.getsize(tmp) == 0:
            print(f"geometry generation produced an empty {path}", flush=True)
            _discard()
            return False
        # Atomic publish.  Until this line the destination still holds whatever
        # it held before, so an unlocked get_wfn() elsewhere cannot read a prefix.
        os.replace(tmp, path)
        return True


    def write_geometry(self, geomval, currGeom, temp = True):
        """Stage temp/<geom>.xyz, the copy every optimisation reads from.

           Write-once, and make_geometry.py is deterministic, so two workers
           racing here produce identical bytes and the atomic publish means a
           reader sees either nothing or the finished file: no lock needed."""
        if temp: 
            path = f"temp/{currGeom}.xyz"
        else: 
            path = f"{currGeom}/geom.xyz"
        if os.path.exists(path):
            return True
        return self._generate_geometry(geomval, path)


    #-----Grid refinement scripts--------------------------------------------------------
    def grid_iteration(self, grid, startGeom, eigs, startEigenpair, coalSearch = True):
        """ Wave function and grid interpolation along <grid> 
                <coalSearch> determines if we set off coalescence searches at each geometry (for A2) 
            For A3 search <startK>, <startKNorm> are provided so that we store the correct hessian eigenvalue, i.e. not that we are finding maxProj 
            to the eigenvalue that set off the search!  

            last point info : stores the [ currGeom, eig, eigvecs, Ks, KNorms] from the last two steps
            return FoundSolns, searchedGeoms, eigs, lastpointInfo 
        """

        self.merged = None
        FoundSolns = False
        searchedGeoms = [ startGeom ]
        wfn = self.get_wfn(startGeom)
        wfn.read_from_disk(startGeom+"/"+self.sol)
        prevK, prevKNorm = compute_singleHessian_projector(wfn, startEigenpair[1])
        prevMetric = wfn.integrals.overlap_matrix().copy()

        # so two lists of lists... 
        # so contains currGeom (geom string), maxproj Hess eigval, maxproj Hess eigvec, currKs, currKNorms (for all eigvecs)
        # only keep the last two points
        lastpointInfo = [ ]

        if len(grid)==0:
            """ Empty interpolation grid """
            return False, None, None, []

        for geomval in grid:
            if not ( self.geomMax  >= geomval >= self.geomMin):
                """ Interpolation grid outside desired bounds"""
                return FoundSolns, searchedGeoms, eigs, lastpointInfo

            currGeom = self.make_geom_name(geomval)
            searchedGeoms.append(currGeom)
            # Geom lock to write geometry and create directory
            if not self.write_geometry(geomval, currGeom):
                print(f"Error failed to write geometry file at {currGeom}")
                break

            # temp=True we wont save any discontinuos solutions
            wfn, converged, continuous, eigsystem =  self.relax_solution(searchedGeoms[-2], currGeom, temp=True, setOffDiscont=False)

            if converged:
                """ Write all the geometries if the wfn converges, leads to smooth PESs """
                with geom_lock(currGeom):
                    os.makedirs(currGeom, exist_ok=True)
                    if not self.write_geometry(geomval, currGeom, temp = False):
                        print("Error failed to write geometry file")
                        try:
                            os.rmdir(currGeom)
                        except:
                            pass
                        break
                    self.new_geoms.append(currGeom)

                    if continuous:
                        merged = self.save_or_merge(currGeom, wfn)
                        if merged is not None: 
                            break

                if (not continuous and self.setOffDiscont):
                    self.register_branches([(wfn, wfn.hess_index[0])], currGeom)
                    break
                elif not continuous: 
                    break         
            else:
                break

            # Passed all checks for convergence, match, etc
            self.storeHessian(eigsystem)
            currKs, currKNorms = compute_hessian_projectors(wfn, self.hess_eigvecs[-1])
            _, MaxProjIndex =  singleMode_hessian_projection(prevK, prevKNorm, currKs, currKNorms, prevMetric)
            eigs.append(self.hess_eigvals[-1][MaxProjIndex])

            # so contains currGeom (geom string), maxproj Hess eigval, maxproj Hess eigvec, currKs, currKNorms (for all eigvecs)
            lastpointInfo.append([currGeom, self.hess_eigvals[-1][MaxProjIndex], self.hess_eigvecs[-1][:,MaxProjIndex], currKs, currKNorms])
            # only keep the last two points... 
            lastpointInfo = lastpointInfo[-2:]

            prevMetric = wfn.integrals.overlap_matrix().copy()
            prevK = currKs[MaxProjIndex]
            prevKNorm = currKNorms[MaxProjIndex]

            if coalSearch:
                FoundSolns, _ = self.coalescence_search(wfn, currGeom)
                if FoundSolns:
                    break

        return FoundSolns, searchedGeoms, eigs, lastpointInfo 


    def post_A3_search(self):
        """ This goes through all the identified modes throughout the """
        FoundSol = False 
        for mode in self.modes:
            # Soft modes stores the soft coupling modes independently
            # Iteratate and refine within in each grid where necessary 
            FoundSol = False

            # Convert soft mode information to np arrays 
            geoms_eigs = []
            evecs = []
            for i in range(len(mode)):
                geoms_eigs.append((mode[i][0], mode[i][1]))
                evecs.append(mode[i][2])
            geoms_eigs = np.array(geoms_eigs, dtype=float)
            evecs = np.array(evecs, dtype=float)

            # Find the lowest magnitude eigenvalue 
            EminInd = np.argmin(np.abs(geoms_eigs[:,1]))
            if EminInd == 0 or EminInd==len(mode)-1:
                """ Lowest at the start or end of the grid, the start has already been searched """
                continue

            # Read in wave function at min. Eig. geometry 
            geom = self.make_geom_name(geoms_eigs[EminInd,0])
            wfn = self.get_wfn(geom)
            wfn.read_from_disk(geom + "/" + self.sol)

            # Identify the target hessian index 
            if geoms_eigs[EminInd, 1] < 0 :
                targetInd = self.HessInd - 1
            else:
                targetInd = self.HessInd + 1
            # Search along the correct mode by perturbing along corresponding evec 
            newwfns, FoundSol = self.search_coalescing_partners(wfn, geom, targetInd,evecs[EminInd, :])
            self.register_branches(newwfns, geom)


            if not FoundSol:
                # No solutions found, there may be solutions found via grid division 
                #Select division left or right 
                leftGrad =  abs((geoms_eigs[EminInd  ,1] - geoms_eigs[EminInd-1,1])/(geoms_eigs[EminInd  ,0] - geoms_eigs[EminInd-1,0]))
                rightGrad = abs((geoms_eigs[EminInd+1,1] - geoms_eigs[EminInd  ,1])/(geoms_eigs[EminInd+1,0] - geoms_eigs[EminInd  ,0]))
                startInterpolate = geom
                startEigenpair   = [ geoms_eigs[EminInd,1], evecs[EminInd,:] ]

                order = [] 
                if leftGrad <= rightGrad:
                    #endInterpolate = self.make_geom_name(geoms_eigs[EminInd-1,0])
                    #endEigenpair = [ geoms_eigs[EminInd-1,1], evecs[EminInd-1,:] ]
                    order = [self.make_geom_name(geoms_eigs[EminInd-1,0]), [ geoms_eigs[EminInd-1,1], evecs[EminInd-1,:] ]]
                else:
                    #endInterpolate = self.make_geom_name(geoms_eigs[EminInd+1,0])
                    #endEigenpair = [ geoms_eigs[EminInd+1,1], evecs[EminInd+1,:] ]
                    order = [self.make_geom_name(geoms_eigs[EminInd+1,0]), [ geoms_eigs[EminInd+1,1], evecs[EminInd+1,:] ]]

                for endInterpolate, endEigenpair in order: 
                    # Refine selected grid, propagate only the desired solution, don't search for coalescences yet
                    searchGeom, searchEigval = self.A3_post_refine_grid(startInterpolate, endInterpolate, startEigenpair, endEigenpair)
                    if searchGeom is None: 
                        continue 

                    # Search Geom tells us the best candidate to search, and searchEigval tells us the eigval corresponding to softMode  
                    wfn = self.get_wfn(searchGeom)
                    wfn.read_from_disk(searchGeom + "/" + self.sol)
                    nEigs, nEvecs, _ = self.analyse_hessian(wfn)
                    zi = np.argsort(np.abs(nEigs-searchEigval)).reshape(-1)[0]

                    # Search along the correct mode, targetInd doesn't change of course 
                    newwfns, FoundSol = self.search_coalescing_partners(wfn, searchGeom, targetInd,nEvecs[:, zi])
                    if len(newwfns)!=0:
                        self.register_branches(newwfns, searchGeom)
                    if FoundSol: 
                        break 
        return FoundSol


    def A3_post_refine_grid(self, startGeom, endGeom, startEigenpair, endEigenpair):
        """ Refine grid for an A3 search """
        searchGeom = None 
        searchEigval = None 

        cGeigs = [startEigenpair[0]]
        coarseGrid = self._grid_between(startGeom, endGeom, self.coarseGrain)

        if len(coarseGrid)!=0:
            # I need a way to refactor this so that I only save geometries that I care about
            _, _, cGeigs, _ =  self.grid_iteration(coarseGrid, startGeom, cGeigs, startEigenpair, coalSearch = False)
            # have a function here to extract the step
            if len(cGeigs)==len(coarseGrid)+1:
                cGeigs = cGeigs + [endEigenpair[0]]
            cGeigs = np.array(cGeigs, dtype=float)
            EigMinInd = np.argmin(np.abs(cGeigs))
            if EigMinInd == 0 or EigMinInd == len(cGeigs)-1:
               pass
            else:
                # Account for lack of start geometry in coarseGrid 
                searchGeom = self.make_geom_name(coarseGrid[EigMinInd-1])
                searchEigval = cGeigs[EigMinInd]
        else:
            """ prevGeom and currGeom stay the same """
            pass

        return searchGeom, searchEigval


    def A2_refine_grid_search(self,igeom, startEigsystem):
        """ Refine grid for an A2 search """
        searchAnything = False
        reachedEnd = False
        FoundSols = False

        prevGeom = self.prop_geoms[igeom]
        currGeom = self.prop_geoms[igeom+1]
        targetGeom = currGeom
        cGeigs = [ startEigsystem[0][np.argsort(np.abs(startEigsystem[0]))[0]] ]
        coarseGrid = self._grid_between(prevGeom, currGeom, self.coarseGrain)
        last2pointsInfo = [] 

        if len(coarseGrid)!=0:
            searchAnything = True
            # Append target geometry
            coarseGrid.append(self.geom_value(currGeom))
            EigminInd = np.argsort(np.abs(startEigsystem[0])).reshape(-1)[0]
            startEigenpair = [ startEigsystem[0][EigminInd], startEigsystem[1][:,EigminInd] ]
            # what is last two points info? 
            FoundSols, searchedGeoms, cGeigs, last2pointsInfo = self.grid_iteration(coarseGrid, prevGeom, cGeigs, startEigenpair, coalSearch = True)
            reachedEnd = (cGeigs is not None
                   and len(cGeigs) == len(coarseGrid) + 1)
            
            if (reachedEnd or FoundSols) or (self.merged is not None):
                "Located coalescing partner, propagated to end of grid or merged with another solution"
                # so return the prevKs and prevKNorms for the next computation, but that if that grid iteration has died ?  
                return FoundSols, reachedEnd, searchAnything, last2pointsInfo[0][3], last2pointsInfo[0][4]
            prevGeom = searchedGeoms[-2]
            currGeom = searchedGeoms[-1]
            fGeigs = [ cGeigs[-1] ]
        else:
            " Go straight to fine grid "
            fGeigs = cGeigs

        fineGrid = list(self._grid_between(prevGeom, currGeom, self.fineGrain))
        if len(fineGrid)==0 and len(coarseGrid)==0:
            # no interpolation done
            print("grid resolution max already - no refinement carried out")
            return False, False, False, [], []
        elif len(fineGrid)!=0: 
            searchAnything = True
            fineGrid.append(self.geom_value(currGeom))
            # StartEigenpair
            if len(last2pointsInfo)!=0: 
                startEigenpair = [ last2pointsInfo[-1][1], last2pointsInfo[-1][2]]
            FoundSols, _, fGeigs, last2pointsInfo = self.grid_iteration(fineGrid, prevGeom, fGeigs, startEigenpair, coalSearch=True)
            reachedEnd = (fGeigs is not None
                   and len(fGeigs) == len(fineGrid) + 1)
            if FoundSols or (self.merged is not None):
                "Located coalescing partner, propagated to end of grid or merged with another solution"
                return FoundSols, reachedEnd, searchAnything, last2pointsInfo[0][3], last2pointsInfo[0][4]
            elif (reachedEnd and not FoundSols):
                " Reached end of the fine grid, but didn't find anything. Needs to go onto the next grid point "
                if currGeom != targetGeom:
                    fineGrid = list(self._grid_between(currGeom, targetGeom, self.fineGrain))
                    fineGrid.append(self.geom_value(targetGeom))
                    fGeigs = [ fGeigs[-1] ]
                    if len(last2pointsInfo)!=0: 
                        startEigenpair = [ last2pointsInfo[-1][1], last2pointsInfo[-1][2]]
                    FoundSols, searchedGeoms, fGeigs, last2pointsInfo = self.grid_iteration(fineGrid, currGeom, fGeigs, startEigenpair, coalSearch = True)
                    reachedEnd = (fGeigs is not None and len(fGeigs) == len(fineGrid) + 1)

        # what are these last two points returning? prevKs and currKs.              
        return FoundSols, reachedEnd, searchAnything, last2pointsInfo[0][3], last2pointsInfo[0][4]


    #----Hessian scripts (store and project)--------------------------------------------------------
    def storeHessian(self, eigsystem):
        self.hess_eigvals.append(eigsystem[0][:self.nhess_elements])
        self.hess_eigvecs.append(eigsystem[1][:,:self.nhess_elements])
        self.hess_eigvals = self.hess_eigvals[-self.nstore_hess:].copy()
        self.hess_eigvecs = self.hess_eigvecs[-self.nstore_hess:].copy()
        return 


    def analyse_hessian(self,wfn): 
        eigval, eigvec, currHessInd = analyse_hessian(wfn, self.hessThresh, exact=self.exactHess)
        return eigval, eigvec, currHessInd 


    def hessian_projection(self, wfn, currGeom, searchInfo, prevNumSoftModes, prevKs, prevKNorms):  
        # This should be able to deal with subspaces of Hessian for larger basis sets. 

        # Then here I should build the modes lists
        if len(searchInfo)> 0: 
            # Then we should construct as save the prev Ks. 
            currKs, currKNorms = compute_hessian_projectors(wfn, self.hess_eigvecs[-1])
            geomVal = self.geom_value(currGeom) 
            if prevNumSoftModes == 0 :
                # This means that prevKs = None, therefore we can just skip this section and start a new mode branch 
                for _, searchModeIndex in searchInfo:
                    self.modes.append([(geomVal, self.hess_eigvals[-1][searchModeIndex], self.hess_eigvecs[-1][:,searchModeIndex])])
            else: 
                metric = wfn.integrals.overlap_matrix() 
                for _, searchModeIndex in searchInfo:
                    approx_proj, maxProjIndex = singleMode_hessian_projection(currKs[searchModeIndex], currKNorms[searchModeIndex], prevKs, prevKNorms, metric) 
                    match = False
                    if 1-np.abs(approx_proj[maxProjIndex])> 2e-1:
                        #print("1-abs(proj): ", 1-np.abs(approx_proj[maxProjIndex]))
                        self.modes.append([(geomVal, self.hess_eigvals[-1][searchModeIndex], self.hess_eigvecs[-1][:,searchModeIndex])]) 
                    else:
                        for mInd, mode in enumerate(self.modes):
                            #mode_geoms = np.array([ x[0] for x in mode ])  
                            #geomIndex = np.argwhere(mode_geoms == self.geom_value(prevGeom))                    
                            if mode[-1][1] == self.hess_eigvals[-2][maxProjIndex]:
                                match = True
                                self.modes[mInd].append((geomVal, self.hess_eigvals[-1][searchModeIndex], self.hess_eigvecs[-1][:,searchModeIndex])) 
                        if not match:
                            self.modes.append([(geomVal, self.hess_eigvals[-1][searchModeIndex], self.hess_eigvecs[-1][:,searchModeIndex])]) 
        
            # Reset for the next iteration 
            prevKs = currKs 
            prevKNorms = currKNorms
            prevNumSoftModes = len(searchInfo)  
        else: 
            # No coalescence signature whatsoever 
            # Reset for next iteration 
            prevNumSoftModes = 0 
            prevKs = [] 
            prevKNorms = [] 
        return prevNumSoftModes, prevKs, prevKNorms 

   
    #---- Register and check solutions----------------------------------------------------------
    def adjacent_geoms(self, geom): 
        allgeoms = sorted(glob.glob("geom_*"), key=self.geom_value)
        vals = [ self.geom_value(g) for g in allgeoms ]
        i = bisect.bisect_left(vals, self.geom_value(geom))
        below = allgeoms[i-1] if i > 0 else None
        if i >= len(allgeoms):
            above = None 
        elif allgeoms[i] == geom and i+1 < len(allgeoms):
            above = allgeoms[i+1]
        elif allgeoms[i] == geom:
            above = None
        else:
            above = allgeoms[i]
        return below, above


    def check_adjacent_geoms(self, geom, wfn, exclude=None): 
        below, above = self.adjacent_geoms(geom)
        match = None   
        for geom_to in [ below , above]:
            if geom_to is None: 
                continue 
            # This introduces a continuity check that might increase cost too much 
            #new_wfn, converged, continuous, _ = self.relax_solution(geom, geom_to, setOffDiscont=False)        
            #if converged and continuous: 
            #    match = match_known_solution(geom_to, new_wfn, self.dedupEnergyThresh, self.dedupOverlapThresh, exclude = exclude)
            #    if match is not None: 
            #        break 
            #else: 
            #    continue 

            continuous = False 
            newwfn = self.get_wfn(geom_to)
            newwfn.initialise(mo_guess = wfn.mo_coeff.copy())
            if not self.PropOPT().run(newwfn, index=self.HessInd, plev=1):
                continue  
            # Check known solutions at geom_to
            _ , _, newInd = self.analyse_hessian(newwfn)
            match = match_known_solution(geom_to, newwfn, self.dedupEnergyThresh, self.dedupOverlapThresh, exclude = exclude)


            # if no matches then skip to next check    
            if match is None: 
                continue  
            else: 
                # Now need back propagation to check continuity
                backwfn = self.get_wfn(geom)
                backwfn.initialise(mo_guess=newwfn.mo_coeff.copy()) 
                if self.PropOPT().run(backwfn, index=self.HessInd, plev=0):
                    if abs(wfn.energy - backwfn.energy) < self.contEnergyThresh:
                        if 1-abs(wfn.overlap(backwfn)) < self.contOverlapThresh:
                            continuous = True
                else: 
                    # then not a true continuous match 
                    match = None 
                    continue
                
                _ , _, backInd = self.analyse_hessian(backwfn)
                continuous = ( continuous and (newInd == self.HessInd) )  
                continuous = ( continuous and (backInd == self.HessInd) )  
 
                if continuous: 
                    # Then match and continuous -> therefore break 
                    break
 
        return match, geom_to              


    def register_branches(self, newwfns, geom):
        """Check, name, save and queue every (wfn, hess_index) partner found at geom."""
        foundSolns = False
        match = []
        #if self.method == "uhf": 
        #    flips = [] 
        #    for nwfn, ind in newwfns: 

        # but also here we may put in a symmetry check! 
        for nwfn, ind in newwfns:
            # check the energy target.
            if self.energy_thresh: 
                if self.energy_thresh < nwfn.energy : 
                    continue 
        
            adj_match, adj_geom = self.check_adjacent_geoms(geom, nwfn, self.sol) 
            if adj_match is not None:
                print(f"  Solution matches with adjacent solution {adj_match} at {adj_geom}") 
                continue
         
            with geom_lock(geom):
                known = match_known_solution(geom, nwfn, self.dedupEnergyThresh,
                                             self.dedupOverlapThresh)
                if known is not None:
                    print(f"  candidate is already-known solution {known}, skipping")
                    match.append(known)
                    continue
                # Visible Provisional name
                prov = f"tmp_{uuid.uuid4().hex}"
                if ind is not None: 
                    nwfn.hess_index = (ind, 0)
                nwfn.save_to_disk(geom + "/" + prov)
                print(f"Found a coalescing partner! -> {prov} at {geom} "
                      f"(index {ind}, E={nwfn.energy:.10f})")
                # be careful of the race condition here
                print("REGISTERING NEW SOLUTIONS!!!!!!", flush=True )
                results = {}
                results["prov"] = prov
                results["geom"] = geom
                results["parent"] = self.sol
                self.proQueue.put(("found", results))
                foundSolns=True
        return foundSolns, match


    #---- Coalescence searches ----------------------------------------------------------------------
    def search_coalescing_partners(self, wfn, geom, search_ind, zero_vec):
        scaler=0.01
        found = []
        FoundCoal = False 
        print("Searching for coalescing partners",flush=True)
        for sign in (-1.00, +1.00):
            testwfn = self.get_wfn(geom) 
            testwfn.initialise(mo_guess = wfn.mo_coeff.copy()) 
            testwfn.take_step(sign * scaler * zero_vec)
            if not HybridEF().run(testwfn, index=search_ind, plev=1, maxit=30,thresh=1e-6): 
                continue 
            if not self.PropOPT().run(testwfn, index=search_ind, thresh=1e-7): 
                continue 

            if any(1-abs(testwfn.overlap(f)) < self.dedupOverlapThresh for f, _ in found):
                print("Matches previous solution")
                continue
            _ , _ , testHessInd = self.analyse_hessian(testwfn)

            # Anything that converged is a solution in its own right and is worth
            # setting off as a branch; only the ones close enough in overlap are
            # the coalescing partner this search was looking for.
            # but some may be close in enough in energy to warrant propagation  
            energy_diff = 27.2114*np.abs(wfn.energy - testwfn.energy)
            if energy_diff < self.FalseCoalEWind and self.setOffFalseCoal:  
                print(f"Energy difference {energy_diff}, new sol. set off ", flush=True)
                found.append((testwfn, testHessInd))
                continue 
             
            ovlpCond = 1-np.abs(wfn.overlap(testwfn))
            print(f"  search candidate at {geom} searchInd={search_ind},sign={sign} scale={scaler} "
                  f"E={testwfn.energy:.10f} index={testHessInd} "
                  f"1-|ovlp|={ovlpCond:.6f}")
                
            if ovlpCond >= self.CoalOverlapThresh:
                # Distant unrelated solution, not the coalescing partner
                print(f"Distant solution at {geom}, not set off (1-abs(ovlp)): ",
                      ovlpCond, flush=True)
                continue
            else: 
                print(f"Found a coalescing partner at {geom} (not checked if new)")
                FoundCoal = True
                found.append((testwfn, testHessInd))
        return found, FoundCoal


    def coalescence_search(self, wfn, geom, prevFoundSol = False, prevNumSoftModes = 0 ):
        """Analyse wfn's Hessian, if a coalescence signature is present, run
           the partner search.
           Returns (newwfns, signature):
                signature = True : when n a zero mode was identified"""
        # Current descriptors 
        FoundCoal = False
        eigval = self.hess_eigvals[-1]
        eigvec = self.hess_eigvecs[-1]

        # It is at this point that we would need to have a check that we are not having just a soft mode! 
        searchInfo = pick_search_index(eigval, self.HessInd, self.CoalSigThresh)
        if len(searchInfo) > 0 :
            if (len(searchInfo) - prevNumSoftModes <= 0): 
                # then we need to analyse the hessian and the eigenvalues for previous steps 
                # i.e that the number of soft modes has changed? 
                pass 
            elif not prevFoundSol: 
                print("Coalesence signature present, no previous coalescing solution located ", flush=True)
                newwfns = [] 
                
                for searchInd, zi in searchInfo:
                    wfns, foundcoal = self.search_coalescing_partners(wfn, geom, searchInd, eigvec[:, zi])
                    # Every converged candidate gets registered; FoundSolns tracks
                    # only whether the coalescing partner itself was located, since
                    # that is what the cusp logic upstream keys off.
                    newwfns += wfns
                    FoundCoal = (FoundCoal or foundcoal)

                if len(newwfns)!=0:
                    self.register_branches(newwfns, geom)
        return FoundCoal, searchInfo 


    #---- Propagation wrappers ----------------------------------------------------------------------
    def relax_solution(self, geom_from, geom_to, temp=False, setOffDiscont = True):
        """Read `sol` at geom_from and re-relax it at geom_to"""
        continuous = False
        
        wfn = self.get_wfn(geom_to, temp)
        wfn.read_from_disk(geom_from + "/" + self.sol)
        converged = self.PropOPT().run(wfn, index=self.HessInd, plev=1)
        if not converged:
            # Didnt even converge - no Hessian was built, so there is no
            # eigensystem to hand back.  Still a 4-tuple: every caller unpacks
            # four, and this is the most travelled exit in the function.
            return wfn, converged, continuous, (None, None)

        # Back propagation to check continuity
        testwfn = self.get_wfn(geom_from)
        testwfn.initialise(mo_guess=wfn.mo_coeff.copy()) 
        if self.PropOPT().run(testwfn, index=self.HessInd, plev=0):
            refwfn = self.get_wfn(geom_from)
            refwfn.read_from_disk(geom_from + "/" + self.sol)
            if abs(refwfn.energy - testwfn.energy) < self.contEnergyThresh:
                if 1-abs(refwfn.overlap(testwfn)) < self.contOverlapThresh:
                    continuous = True
                else:
                    print("Discontinuity: failed 1-abs(ovlp) : ", 1-abs(refwfn.overlap(testwfn)))
            else:
                print("Discontinuity: energy too far: ", refwfn.energy, testwfn.energy,
                      " dE=", abs(refwfn.energy - testwfn.energy))

        # So this is where some eigenfollow logic would hold us back 
        if converged:
            eigval, eigvec, currHessInd = self.analyse_hessian(wfn)
            #report_hessian(geom_to, self.HessInd, eigval, currHessInd)
            continuous = ( continuous and (currHessInd == self.HessInd) )  
            
        if self.setOffDiscont and ( (not temp and setOffDiscont) and (converged and not continuous)):
            # This sets off as another branch any new solutions  
            print("Not continuous, setting off new branch", flush=True ) 
            self.register_branches([(wfn, currHessInd)], geom_to)
        return wfn, converged, continuous, [eigval[:self.nhess_elements], eigvec[:,:self.nhess_elements]]


    def propagate_solution(self, initSearch=False):
        print("Starting propagations: sol, prop_geoms ", self.sol, self.prop_geoms)
        startE, _ = read_solution_header(
            self.prop_geoms[0] + "/" + self.sol + ".solution")

        self.Energies = [startE]
        prevFoundSolns = False
    
        prevKs = []
        prevKNorms = []
        prevNumSoftModes = 0
        
        for igeom, geom in enumerate(self.prop_geoms[1:]):
            print(f"====== Geom: {geom} ========")  
            prev_geom = self.prop_geoms[igeom]
            wfn, converged, continuous, eigsystem = self.relax_solution(prev_geom, geom, temp=False, setOffDiscont = True)
            print(f"Relax output: converged: {converged} and continuous: {continuous}")  
            merged = None
            # Becareful of this condition and if the optimisation doesnt converge
            # This is fine and doesn't need refactoring
            if (not converged or not continuous):
                print("Current solution is being trashed/(set off again) - can we subdivide geometry steps?")
                if prevFoundSolns:
                    print("PES died but we have located previous coalescing solution")
                    break

                currFoundSolns, reachedEnd, searchedAnything, prevKs, prevKNorms = self.A2_refine_grid_search(
                    igeom, (self.hess_eigvals[-1], self.hess_eigvecs[-1]))

                print(f" A2 Fold search: reachedEnd: {reachedEnd}, FoundSolns: {currFoundSolns} and searchedAnything: {searchedAnything}")  
                if not reachedEnd or (self.merged is not None): 
                    break 
                else:  
                    # The refinement bridged the interval and saved self.sol at
                    # geom, so the step was only too coarse.  Adopt that point
                    # what if it was a previously converged onto solution? 
                    wfn.read_from_disk(geom + "/" + self.sol)
                    eigval, eigvec, currHessInd = self.analyse_hessian(wfn)
                    eigsystem = [ eigval, eigvec] 
                    prevFoundSolns = False
                    self.storeHessian(eigsystem)
                    self.Energies.append(wfn.energy)
                    # But we need to remake the prevKs, prevKNorms 
                    #prevNumSoftModes, prevKs, prevKNorms = self.hessian_projection(wfn, geom, searchInfo, prevNumSoftModes, prevKs, prevKNorms)  
                    continue  

            with geom_lock(geom):
                merged = self.save_or_merge(geom, wfn)
                if merged is not None: 
                        break 
            if igeom == 0: 
                # Then we should check the start for coalesence signature  
                refwfn = self.get_wfn(self.prop_geoms[0])
                refwfn.read_from_disk(self.prop_geoms[0]+"/"+self.sol)
                if initSearch:
                    refsearchInfo = pick_search_index(self.hess_eigvals[0], self.HessInd, self.CoalSigThresh)
                    print("Init. Search Info, ", refsearchInfo)  
                    print("Initial eigenvalues: ", self.hess_eigvals[0][:10])
                    if len(refsearchInfo) > 0:
                        refnewwfns = []
                        for refsearchInd, refzi in refsearchInfo:
                            refwfns, _ = self.search_coalescing_partners(refwfn, self.prop_geoms[0], refsearchInd, self.hess_eigvecs[0][:,refzi])
                            refnewwfns += refwfns
                        if len(refnewwfns)!=0 :
                            print("Potentially new solutions from initial coalescence search")
                            self.register_branches(refnewwfns, self.prop_geoms[0])
           
            self.storeHessian(eigsystem)
            self.Energies.append(wfn.energy) 
            # So this is going to be where we decide to set off a search etc
            prevFoundSolns, searchInfo  = self.coalescence_search(wfn, geom,
                                                              prevFoundSol = prevFoundSolns,
                                                              prevNumSoftModes = prevNumSoftModes)
            prevNumSoftModes, prevKs, prevKNorms = self.hessian_projection(wfn, geom, searchInfo, prevNumSoftModes, prevKs, prevKNorms)  
        return

 
    def initialise_run(self, geom):
        """Initial search for coalescing solutions"""
        wfn = self.get_wfn(geom)
        wfn.read_from_disk(geom+"/"+self.sol)
        # so here we can set the class wide settings for kwargs
        if self.method=="csf": 
            self.spin_coupling = wfn.spin_coupling 
        else: 
            pass 
         
        wfn.update()
        eigval, eigvec, currHessInd = self.analyse_hessian(wfn)
        self.HessInd = currHessInd
        self.nhess_elements = currHessInd + 4  
        self.storeHessian([eigval, eigvec])
        
        # so I could/should put the search/index run after doing this? 
        #if search:
        #    searchInfo = pick_search_index(eigval,currHessInd, self.CoalSigThresh)
        #    print("Search Info, ", searchInfo)  
        #    print("Initial eigenvalues: ", eigval[:10]) 
        #    if len(searchInfo) > 0:
        #        newwfns = []
        #        for searchInd, zi in searchInfo:
        #            wfns, _ = self.search_coalescing_partners(wfn, geom, searchInd, eigvec[:,zi])
        #            newwfns += wfns
        #        if len(newwfns)!=0 :
        #            print("Potentially new solutions from initial coalescence search")
        #            self.register_branches(newwfns, geom)
        return


    def construct_PES(self, start_geom, forward=True, initSearch=True):
        self._check_in(f"{'fwd' if forward else 'bwd'}")
        # Find the propagation geometries
        ind = self.geoms.index(start_geom)
        if forward:
            self.prop_geoms = self.geoms[ind:]
        else:
            self.prop_geoms = self.geoms[:ind + 1][::-1]

        self.initialise_run(self.prop_geoms[0])
        if len(self.prop_geoms)>1:
            # We only run the initial coalescence search if we survive the first step 
            self.propagate_solution(initSearch=initSearch)
        print(f"{self.sol} finished, geometries created this walk: ", self.new_geoms)

        print("Performing A3 search") 
        print("soft modes: ",self.modes )
        self.post_A3_search()
        return self.new_geoms.copy()


    def fill_solution_gaps(self):
        """Interior gap-fill: put self.sol onto every final-grid point inside the
           span it already covers but is currently missing from.

            This will run in a new set of calculations, recall self.geoms is the entire list
        """
        self._check_in("sweep")
        present = sorted(
            (g for g in self.geoms if os.path.exists(f"{g}/{self.sol}.solution")),
            key=self.geom_value)
        if len(present) < 2:
            return
        start, stop = self.geom_value(present[0]), self.geom_value(present[-1])
        present_set = set(present)

        # HessInd of this solution (from its stored header) drives the optimizer
        _, self.HessInd = read_solution_header(f"{present[0]}/{self.sol}.solution")

        # march inward from each existing anchor so consecutive holes chain guesses
        for i in range(len(present) - 1):
            anchor = present[i]
            # final-grid points strictly between present[i] and present[i+1]
            holes = [g for g in self.geoms
                     if self.geom_value(present[i]) < self.geom_value(g) < self.geom_value(present[i+1])]
            for hole in sorted(holes, key=self.geom_value):
                wfn, converged, continuous, _ = self.relax_solution(anchor, hole, setOffDiscont=False)
                if not (converged and continuous):
                    print(f"  {self.sol}: gap at {hole} is a REAL discontinuity, not filling")
                    continue
                with geom_lock(hole):
                    _ = self.save_or_merge(hole, wfn)

                anchor = hole   # chain: next hole relaxes from the one we just filled


    def save_or_merge(self, geomName, wfn, claim = True):
        # Claim just says if we register a claim or not 
        merged = None 
        merged = match_known_solution(geomName, wfn, self.dedupEnergyThresh,
                                                          self.dedupOverlapThresh, exclude = self.sol)
        if (merged is None):
            wfn.save_to_disk(geomName + "/" + self.sol)
        elif claim: 
            self.merged = merged
            self.proQueue.put(("merged", {"tid": self.tid, "sol": self.sol,
                                     "onto": merged, "geom": geomName}))
            print(f"{self.sol} converged onto known solution {merged} at {geomName}")
        return merged