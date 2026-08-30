#!/usr/bin/python3
from quantel.ints.pyscf_integrals import PySCFIntegrals, PySCFMolecule
from quantel.opt.eigenvector_following import EigenFollow
from quantel.opt.hybrid_ef import HybridEF
from quantel.opt.lsr1 import SR1
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
        if config["wfn"]=="uhf":
            from quantel.wfn.uhf import UHF
            self.WFN = UHF
        elif config["wfn"]=="csf": 
            from quantel.wfn.csf import CSF
            self.WFN = CSF  
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
        self.hess_eigvals = []
        self.hess_eigvecs = []
        # Child process selection  
        self.setOffDiscont = config["jobcontrol"]["setOffDiscount"] 
        self.setOffFalseCoal = config["jobcontrol"]["setOffFalseCoal"]

        # Geometry information
        self.geomMax = config["geoms"]["geomMax"] 
        self.geomMin = config["geoms"]["geomMin"]
        self.includeSign = config["geoms"]["includeSign"]
        self.leading_zeros  = config["geoms"]["leading_zeros"]
        self.geomDecimals = config["geoms"]["geomDecimals"]
        self.fineGrain = 10**(- self.geomDecimals) 
        self.coarseGrain = config["geoms"]["coarseGrain"]
        self.makeGeomscript = config["geoms"]["makeGeomscript"] 
        self.new_geoms = []

        # To assign the hessian index
        self.hessThresh = 1e-16
        # Deduplication thresholds 
        self.dedupEnergyThresh = 1e-8          
        self.dedupOverlapThresh = 1e-5    
        # Threshold for continuity backtracking to current solution 
        self.contEnergyThresh  = 1e-4     
        self.contOverlapThresh = 1e-4     

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
        return self.WFN(ints, "+-")

    #-------------------------------------------------------------
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
        """Interior grid values strictly between the two geometries, in walk
           order, clipped to [geomMin, geomMax].

           The number of steps is counted first rather than handing the bounds
           to np.arange: (2.6-2.5)/0.1 is 1.0000000000000009 in binary, so
           arange sizes the array one element too big and [1:] hands back the
           endpoint as though it were an interior point.  On a grid already at
           the naming resolution that made 40% of intervals report a point to
           refine, and the refinement then re-ran the step it had just failed."""
        if step < 10 **(-self.geomDecimals):
            raise ValueError(
                f"grid step {step} is finer than the {self.geomDecimals}-decimal "
                f"geometry naming; distinct points would collide on one directory")
        a, b = self.geom_value(from_geom), self.geom_value(to_geom)
        nstep = int(round(abs(b - a) / step))
        sign = 1.0 if a < b else -1.0
        # Rounded onto the naming grid so make_geom_name and geom_value
        # round-trip exactly instead of drifting by an accumulated ulp.  The
        # +0.0 normalises the negative zero a descending walk lands on, which
        # would otherwise be named geom_-0.0 - a second directory for a
        # geometry already held as geom_00.0.
        grid = np.round(a + sign * step * np.arange(1, nstep), self.geomDecimals) + 0.0
        grid = [ x for x in grid if  self.geomMax >= x >= self.geomMin ]
        return grid
    #-------------------------------------------------------------

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

    def write_geometry(self, geomval, currGeom):
        """Stage temp/<geom>.xyz, the copy every optimisation reads from.

           Write-once, and make_geometry.py is deterministic, so two workers
           racing here produce identical bytes and the atomic publish means a
           reader sees either nothing or the finished file: no lock needed."""
        path = f"temp/{currGeom}.xyz"
        if os.path.exists(path):
            return True
        return self._generate_geometry(geomval, path)

    def commit_geometry(self, geomval, currGeom):
        """Publish <geom>/geom.xyz, once something worth keeping has converged
           at this geometry.

           Regenerated rather than copied from the staged file - same script,
           same value, so the same coordinates this wavefunction was built
           from.  Until this runs the directory does not exist, so a point the
           walk gave up on leaves nothing behind to clean away."""
        path = f"{currGeom}/geom.xyz"
        if os.path.exists(path):
            return True
        return self._generate_geometry(geomval, path)
    
    def grid_iteration(self, grid, prevGeom, eigs):
        self.merged = None 
        FoundSolns = False
        searchedGeoms = [ prevGeom ] 
        if len(grid)==0:
            return False, None, None  
        # We also want the first and last hessian eigenvalues  
        #hess_eigvals = [starteig, ...., endeig]
        for geomval in grid:
            # New addition of code..
            if not ( self.geomMax  >= geomval >= self.geomMin): 
                return FoundSolns, searchedGeoms, eigs 
                
            currGeom = self.make_geom_name(geomval)
            print("In grid, currGeom", currGeom)  
            searchedGeoms.append(currGeom) 
            # Geom lock to write geometry and create directory 
            if not self.write_geometry(geomval, currGeom):
                print("Error failed to write geometry file") 
                break  
            print("prevGeom, currGeom relax :", searchedGeoms[-2], currGeom, flush=True)  
            # since temp=True we wont save any discontinuos solutions  
            wfn, converged, continuous, eigsystem =  self.relax_solution(searchedGeoms[-2], currGeom, temp=True, setOffDiscont=False)
            
            if converged: 
                with geom_lock(currGeom):
                    os.makedirs(currGeom, exist_ok=True)
                    if not self.commit_geometry(geomval, currGeom):
                        print("Error failed to write geometry file")
                        try:     
                            os.rmdir(currGeom) 
                        except: 
                            pass  
                        break 
                    self.new_geoms.append(currGeom)
                    
                    if continuous:
                        merged = match_known_solution(currGeom, wfn, self.dedupEnergyThresh,
                                                      self.dedupOverlapThresh, exclude=self.sol)
                        if (merged is None):
                            wfn.save_to_disk(currGeom + "/" + self.sol)
                        else:
                            self.merged = merged 
                            print(f"{self.sol} converged onto known solution {merged} at {currGeom}")
                            self.merged = merged
                            self.proQueue.put(("merged", {"tid": self.tid, "sol": self.sol,
                                                 "onto": merged, "geom": currGeom}))
                            break 
                
                if (not continuous and self.setOffDiscont):  
                    self.register_branches([(wfn, wfn.hess_index[0])], currGeom)
                    break 
            else: 
                break  
            
            self.hess_eigvals.append(eigsystem[0])
            self.hess_eigvecs.append(eigsystem[1])
            self.hess_eigvals = self.hess_eigvals[-self.nstore_hess:].copy()
            self.hess_eigvecs = self.hess_eigvecs[-self.nstore_hess:].copy()
            FoundSolns, _, eigval = self.coalescence_search(wfn, currGeom) 
            if FoundSolns: 
                break 
            eigs.append(eigval[np.argsort(np.abs(eigval))[0]]) 
                
        return FoundSolns, searchedGeoms, eigs  
    
    def _estimate_cusp(self, eigvals, names, targetGeom): 
        # Catch when mInd is either the [0] or [-1] value of the coarse grid even after splitting  
        mInd = np.argmin(np.abs(eigvals))
        if mInd == 0:  
            # if mInd is zero that means we need to consider this against the previous coarse grain interval
            # or that we should just divide the first section...  
            prevGeom, currGeom = names[mInd], names[mInd+1]
            fGeigs = [ eigvals[mInd] ] 
        elif mInd == len(eigvals)-1: 
            # if mInd is -1 then means we need to check against the next coarse grain interval...
            addGeom = self.geom_value(targetGeom)+(self.geom_value(names[-1]) - self.geom_value(names[-2])) 
            addEigs = [eigvals[-2]] 
            FoundSolns, _, addEigs =  self.grid_iteration([addGeom], names[mInd], addEigs)
            if FoundSolns or (self.merged is not None): 
                return None, None, None, True 
            # So checking if is better to move on to search on other side or to search here
            j = int(np.argmin(np.abs(np.array(addEigs)-eigvals[mInd])))
            if j == 1:
                return None, None, None, False  
            prevGeom, currGeom = names[mInd-1], names[mInd]
            fGeigs = [ eigvals[mInd-1] ] 
              
        else:
            # See which is the smallest drop out of the forward and backward change  
            j = int(np.argmin(np.abs([eigvals[mInd]-eigvals[mInd-1], eigvals[mInd+1]-eigvals[mInd]])))
            prevGeom, currGeom = names[mInd-1+j], names[mInd+j]
            fGeigs = [ eigvals[mInd-1+j] ] 
        return prevGeom, currGeom, fGeigs, False      


    def refine_grid_search(self,igeom,savedEigvals, search="A2"): 
        """ Refine grid for an A2 search """
        searchAnything = False
        reachedEnd = False
        FoundSolns = False
     
        prevGeom = self.prop_geoms[igeom]
        currGeom = self.prop_geoms[igeom+1]
        targetGeom = currGeom
        cGeigs = [ savedEigvals[0][np.argsort(np.abs(savedEigvals[0]))[0]] ]
        coarseGrid = self._grid_between(prevGeom, currGeom, self.coarseGrain)
        if len(coarseGrid)!=0:
            searchAnything = True  
            # Append target geometry
            coarseGrid.append(self.geom_value(currGeom))
            print("A2 refine coarseGrid: ", coarseGrid) 
            FoundSolns, searchedGeoms, cGeigs = self.grid_iteration(coarseGrid, prevGeom, cGeigs)
            reachedEnd = (cGeigs is not None
                   and len(cGeigs) == len(coarseGrid) + 1)
            if ((search=="A2") and (reachedEnd or FoundSolns)) or (self.merged is not None):
                "Located coalescing partner, propagated to end of grid or merged with another solutoin" 
                return FoundSolns, reachedEnd, searchAnything  

            if search=="A2": 
                prevGeom = searchedGeoms[-2]
                currGeom = searchedGeoms[-1]
                fGeigs = [ cGeigs[-1] ] 
            elif search=="A3": 
                # have a function here to extract the step 
                eigvals = np.asarray(cGeigs, dtype=float)
                names = searchedGeoms[:len(eigvals)]
                if len(names) < 2 : 
                    currGeom = searchedGeoms[-1]
                    fGeigs = [ cGeigs[-1] ] 
                else:
                    prevGeom, currGeom, fGeigs, FoundSolns = self._estimate_cusp(eigvals, names, targetGeom ) 
                    if (prevGeom is None) or (FoundSolns or (self.merged is not None)): 
                        return FoundSolns, reachedEnd, searchAnything  
        else: 
            " prevGeom and currGeom stay the same "
            fGeigs = cGeigs
        
        fineGrid = list(self._grid_between(prevGeom, currGeom, self.fineGrain))
        if len(fineGrid)!=0:
            searchAnything = True  
            fineGrid.append(self.geom_value(currGeom))
            FoundSolns, _, fGeigs = self.grid_iteration(fineGrid, prevGeom,
                                                    fGeigs)
            reachedEnd = (fGeigs is not None
                   and len(fGeigs) == len(fineGrid) + 1)
            if (reachedEnd and not FoundSolns) and (self.merged is None): 
                " well this is a challenging case - if may propagate along current trajectory "
                if currGeom != targetGeom: 
                    fineGrid = list(self._grid_between(currGeom, targetGeom, self.fineGrain))
                    fineGrid.append(self.geom_value(targetGeom)) 
            
                    fGeigs = [ fGeigs[-1] ] 
                    FoundSolns, _, fGeigs = self.grid_iteration(fineGrid, currGeom,
                                                           fGeigs)
                    reachedEnd = (fGeigs is not None
                           and len(fGeigs) == len(fineGrid) + 1)
        return FoundSolns, reachedEnd, searchAnything     

    def old_refine_grid_search(self, igeom, savedEigvals):
        """Subdivide the interval the main walk just failed to cross.
           or to locate A3 cusp 
            """
        prevGeom = self.prop_geoms[igeom]
        currGeom = self.prop_geoms[igeom+1]
        # Plain rebind, not .copy(): prop_geoms mixes np.str_ (from geoms.txt)
        # with plain str (every point merge_geom inserts mid-run), and str has
        # no .copy().  Names are immutable either way.
        targetGeom = currGeom
        cGeigs = [ savedEigvals[0][np.argsort(np.abs(savedEigvals[0]))[0]] ]
        #------------------------
        cuspSearch=None  
        if savedEigvals[1] is None: 
            cuspSearch = 2  
        else: 
            cuspSearch = 3 
        #------------------------

        coarseGrid = self._grid_between(prevGeom, currGeom, self.coarseGrain)
        coarseGrid = list(coarseGrid) 
        # Adding Current geom this'll make it work better - but we have already run this calculation though.. 
        if len(coarseGrid)==0: 
            FoundSolns = False 
            searchedGeoms = [] 
        else: 
            coarseGrid.append(self.geom_value(currGeom))
            print("in refine search coarseGrid: ", coarseGrid) 
            FoundSolns, searchedGeoms, cGeigs  = self.grid_iteration(coarseGrid, prevGeom, cGeigs)
        """
            We want to distinguish between two scenarios
            1. A3 cusp is to be found in a finer grid in this interval not the last gridpoint  
            2. A3 cusp is actually between the next geometry. This stop the fine grid search if the eigenvalues are monotonic  
        """

        # cGeigs = [ prevEig, convergedEigs... ] 
        # searchedGeoms = [ prevGeom, all_attempted Geoms (so includes prevGeom and currGeom eigs always)) 
        reached = (cGeigs is not None
                   and len(cGeigs) == len(coarseGrid) + 1)
        if cuspSearch == 2 and reached:
            print(f"refinement reached {targetGeom} on the coarse grid - "
                  f"propagation can continue", flush=True)
            return FoundSolns, True
        

        if FoundSolns:
            # Found solns and ended early - either A3 or A2 
            return True, False

        if len(searchedGeoms) < 2:
            # Well this should just set off a fine grid calc? 
            # If interval was too small, since coarseGrid has appended currGeom - searchGeom should always have these two 
            return False, False
        
        if cuspSearch == 2:
            prevGeom = searchedGeoms[-2]
            currGeom = searchedGeoms[-1]
            fGeigs = [ cGeigs[-1] ] 
        elif cuspSearch == 3:
            eigvals = np.asarray(cGeigs, dtype=float)
            names = searchedGeoms[:len(eigvals)]
            if len(names) < 2:
                return False, False
            # How to catch when mInd is either the 0 or -1 ? 
            mInd = np.argmin(np.abs(eigvals))
            if mInd == 0:  
                # if mInd is zero that means we need to consider this against the previous coarse grain interval
                # or that we should just divide the first section...  
                prevGeom, currGeom = names[mInd], names[mInd+1]
                fGeigs = [ cGeigs[mInd] ] 
            elif mInd == len(eigvals)-1: 
                # if mInd is -1 then means we need to check against the next coarse grain interval...
                addGeom = self.geom_value(targetGeom)+(self.geom_value(names[-1]) - self.geom_value(names[-2])) 
                addEigs = [cGeigs[-2]] 
                FoundSolns, _, addEigs =  self.grid_iteration([addGeom], names[mInd], addEigs)
                if FoundSolns: 
                    return True, True 
                j = int(np.argmin(np.abs(np.array(addEigs)-cGeigs[mInd])))
                if j == 1: 
                    return FoundSolns,True   
                prevGeom, currGeom = names[mInd-1], names[mInd]
                fGeigs = [ cGeigs[mInd-1] ] 
                  
            else: 
                j = int(np.argmin(np.abs([eigvals[mInd]-eigvals[mInd-1], eigvals[mInd+1]-eigvals[mInd]])))
                prevGeom, currGeom = names[mInd-1+j], names[mInd+j]
                fGeigs = [ cGeigs[mInd-1+j] ] 

        fineGrid = list(self._grid_between(prevGeom, currGeom, self.fineGrain))
        fineGrid.append(self.geom_value(currGeom))
        FoundSolns, _, fGeigs = self.grid_iteration(fineGrid, prevGeom,
                                                    fGeigs)
        if (cuspSearch == 2 and not FoundSolns and currGeom == targetGeom
                and fGeigs is not None
                and len(fGeigs) == len(fineGrid) + 1):
            print(f"refinement reached {targetGeom} on the fine grid - "
                  f"propagation can continue", flush=True)
            return False, True
        return FoundSolns, False

    #-------------------------------------------------------------

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
        testwfn.initialise(mo_guess = wfn.mo_coeff.copy())
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
            eigval, eigvec, currHessInd = analyse_hessian(wfn, self.hessThresh)
            report_hessian(geom_to, self.HessInd, eigval, currHessInd)
            continuous = ( continuous and (currHessInd == self.HessInd) )  
            
        if self.setOffDiscont and ( (not temp and setOffDiscont) and (converged and not continuous)):
            # This sets off as another branch any new solutions  
            print("Not continuous, setting off new branch", flush=True ) 
            self.register_branches([(wfn, currHessInd)], geom_to)
        return wfn, converged, continuous, (eigval, eigvec) 


    def search_coalescing_partners(self, wfn, geom, search_ind, zero_vec):
        scaler=0.01
        found = []
        FoundCoal = False 
        print("Searching for coalescing partners",flush=True)
        for sign in (-1.00, +1.00):
            testwfn = self.get_wfn(geom) 
            testwfn.initialise(mo_guess=wfn.mo_coeff) 
            testwfn.take_step(sign * scaler * zero_vec)
            print(" Hybrid EF max steps = 30 ") 
            if not HybridEF().run(testwfn, index=search_ind, plev=1, maxit=30): 
                continue 
            print(" Testing second portion to push to further tolerances on search") 
            if not self.PropOPT().run(testwfn, index=search_ind): 
                continue 
            teval, _ = np.linalg.eigh(testwfn.hessian)
            testHessInd = np.sum(teval < -self.hessThresh)
            testwfn.hess_index = (testHessInd, 0)
            ovlp = wfn.overlap(testwfn)
            print(f"  search candidate at {geom} searchInd={search_ind},sign={sign} scale={scaler} "
                  f"E={testwfn.energy:.10f} index={testHessInd} "
                  f"|ovlp|={np.abs(ovlp):.6f}")
            
            if any(1-abs(testwfn.overlap(f)) < self.dedupOverlapThresh
                   for f, _ in found):
                print("Failed matches previous solution")
                continue

            # Anything that converged is a solution in its own right and is worth
            # setting off as a branch; only the ones close enough in overlap are
            # the coalescing partner this search was looking for.
            if self.setOffFalseCoal: 
                found.append((testwfn, testHessInd))
                if 1 - np.abs(ovlp) >= self.CoalOverlapThresh:
                    # Distant unrelated solution, not the coalescing partner
                    print(f"Distant solution at {geom} (1-abs(ovlp)): ",
                          1 - np.abs(ovlp), flush=True)
                    continue

                print(f"Found a coalescing partner at {geom} (not checked if new)")
                FoundCoal = True
            else: 
                if 1 - np.abs(ovlp) >= self.CoalOverlapThresh:
                    # Distant unrelated solution, not the coalescing partner
                    print(f"Distant solution at {geom} (1-abs(ovlp)): ",
                          1 - np.abs(ovlp), flush=True)
                    continue

                print(f"Found a coalescing partner at {geom} (not checked if new)")
                FoundCoal = True
                found.append((testwfn, testHessInd))
        return found, FoundCoal

    def coalescence_search(self, wfn, geom, prevFS = False, prevCS = False):
        """Analyse wfn's Hessian, if a coalescence signature is present, run
           the partner search.
           Returns (newwfns, signature):
                signature = True : when n a zero mode was identified"""
        FoundSolns = False 
        CoalSig = False 
        eigval = self.hess_eigvals[-1]
        eigvec = self.hess_eigvecs[-1]
        searchInfo = pick_search_index(eigval, self.HessInd, self.CoalSigThresh)
            
        if len(searchInfo) > 0 :
            CoalSig = True
            if prevFS and prevCS:
                return FoundSolns, CoalSig, eigval 
 
            print("Coalesence signature present ", flush=True)
            newwfns = [] 
            for searchInd, zi in searchInfo:
                wfns, FoundCoal = self.search_coalescing_partners(wfn, geom, searchInd,
                                                 eigvec[:, zi])
                # Every converged candidate gets registered; FoundSolns tracks
                # only whether the coalescing partner itself was located, since
                # that is what the cusp logic upstream keys off.
                newwfns += wfns
                FoundSolns = (FoundSolns or FoundCoal)

            if len(newwfns)!=0:
                self.register_branches(newwfns, geom)
        return FoundSolns, CoalSig, eigval  

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
        _ , _, refInd = analyse_hessian(wfn, self.hessThresh)
        below, above = self.adjacent_geoms(geom)
        match = None   
        for geom_to in [ below , above]:
            if geom_to is None: 
                continue 
            continuous = False 
            # The problem with this is if the spin coupling patterns are different! 
            t1wfn = self.get_wfn(geom_to)
            t1wfn.initialise(mo_guess = wfn.mo_coeff.copy())  
            if not self.PropOPT().run(t1wfn, index=refInd, plev=1):
                continue  
            #match known solutions at geom_to - if it doesnt then we dont need the discont check, if it does we need discont check!   
            _ , _, t1Ind = analyse_hessian(t1wfn, self.hessThresh)
            match = match_known_solution(geom_to, t1wfn, self.dedupEnergyThresh, self.dedupOverlapThresh, exclude = exclude)
            if match is None: 
                continue  
            # Back propagation to check continuity
            t2wfn = self.get_wfn(geom)
            t2wfn.initialise(mo_guess = t1wfn.mo_coeff.copy())
            if self.PropOPT().run(t2wfn, index=refInd, plev=0):
                if abs(wfn.energy - t2wfn.energy) < self.contEnergyThresh:
                    if 1-abs(wfn.overlap(t2wfn)) < self.contOverlapThresh:
                        continuous = True
            else: 
                match = None 
                continue 
            if (refInd != t1Ind) or not continuous: 
                match = None
            
            if match is not None: 
                break 
 
        return match, geom_to              

    def register_branches(self, newwfns, geom):
        """Check, name, save and queue every (wfn, hess_index) partner found at geom."""
        foundSolns = False
        match = []
        for nwfn, ind in newwfns:
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

    def propagate_solution(self, initSearch=False):
        print("Starting propagations: sol, prop_geoms ", self.sol, self.prop_geoms)
        startE, _ = read_solution_header(
            self.prop_geoms[0] + "/" + self.sol + ".solution")

        self.Energies = [startE]
        prevFoundSolns = False
        prevCoalSig = False
        for igeom, geom in enumerate(self.prop_geoms[1:]):
            print(f"====== Geom: {geom} ========")  
            prev_geom = self.prop_geoms[igeom]
            wfn, converged, continuous, eigsystem = self.relax_solution(prev_geom, geom, temp=False, setOffDiscont = True)
            print(f"Relax output: converged: {converged} and continuous: {continuous}")  
            merged = None
            # Becareful of this condition and if the optimisation doesnt converge
            if (not converged or not continuous):
                print("Current solution is being trashed/(set off again) - can we subdivide geometry steps?")
                if prevFoundSolns:
                    print("PES died but we have located previous coalescing solution")
                    break

                currFoundSolns, reachedEnd, searchedAnything = self.refine_grid_search(
                    igeom, (self.hess_eigvals[-1], None), search="A2")

                print(f" A2 Fold search: reachedEnd: {reachedEnd}, FoundSolns: {currFoundSolns} and searchedAnything: {searchedAnything}")  
                if not reachedEnd or (self.merged is not None): 
                    break 
                else:  
                    # The refinement bridged the interval and saved self.sol at
                    # geom, so the step was only too coarse.  Adopt that point
                    # what if it was a previously converged onto solution? Well then... 
                    wfn.read_from_disk(geom + "/" + self.sol)
                    eigval, eigvec, currHessInd = analyse_hessian(wfn, self.hessThresh)
                    eigsystem = [ eigval, eigvec] 
                    prevFoundSolns = False
                    prevCoalSig = False
                    self.hess_eigvals.append(eigsystem[0])
                    self.hess_eigvecs.append(eigsystem[1])
                    self.hess_eigvals = self.hess_eigvals[-self.nstore_hess:].copy()
                    self.hess_eigvecs = self.hess_eigvecs[-self.nstore_hess:].copy()
                    self.Energies.append(wfn.energy)
                    continue  

            with geom_lock(geom):
                merged = match_known_solution(geom, wfn, self.dedupEnergyThresh,
                                              self.dedupOverlapThresh, exclude=self.sol)
                if (merged is None):
                    wfn.save_to_disk(geom + "/" + self.sol)
            

            if merged is not None:
                print(f"{self.sol} converged onto known solution {merged} at {geom}")
                # Report the identity and stop walking; renaming here would pull
                # files out from under the twin walk (which is still seeded from
                # prop_geoms[0]) and out from under any other task part-way
                # through saved_solutions_at() at these geometries.
                self.merged = merged
                self.proQueue.put(("merged", {"tid": self.tid, "sol": self.sol,
                                         "onto": merged, "geom": geom}))
                break
            elif igeom == 0: 
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
           
            self.hess_eigvals.append(eigsystem[0])
            self.hess_eigvecs.append(eigsystem[1])
            self.hess_eigvals = self.hess_eigvals[-self.nstore_hess:].copy()
            self.hess_eigvecs = self.hess_eigvecs[-self.nstore_hess:].copy()
            self.Energies.append(wfn.energy) 
            FoundSolns, CoalSig, eigval = self.coalescence_search(wfn, geom,
                                                             prevFS = prevFoundSolns,
                                                              prevCS = prevCoalSig)
            #  
            if (CoalSig and not FoundSolns) and (prevCoalSig and not prevFoundSolns): 
                print("A3 search: No coalescing partners were located")
                FoundSolns, _ , searchAnything = self.refine_grid_search(
                    igeom, 
                    (self.hess_eigvals[-2], self.hess_eigvals[-1]), search="A3")
                if FoundSolns:
                    print("A3 - constructed new geometries and located a coalescing partners")
                if self.merged is not None: 
                    print("Merged during A3 search but not merged at current geom") 
    
            prevCoalSig = CoalSig 
            if not CoalSig: 
                prevFoundSolns = False 
            else: 
                prevFoundSolns = (prevFoundSolns or FoundSolns) 
                pass  
        return


    def initialise_run(self, geom, search=True):
        """Initial search for coalescing solutions"""
        wfn = self.get_wfn(geom)
        wfn.read_from_disk(geom+"/"+self.sol)
        wfn.update()
        eigval, eigvec, currHessInd = analyse_hessian(wfn, self.hessThresh)
        self.hess_eigvals.append(eigval)
        self.hess_eigvecs.append(eigvec)
        self.HessInd = currHessInd

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


    def construct_PES(self, start_geom, forward=True):
        self._check_in(f"{'fwd' if forward else 'bwd'}")
        # Find the propagation geometries
        ind = self.geoms.index(start_geom)
        if forward:
            self.prop_geoms = self.geoms[ind:]
        else:
            self.prop_geoms = self.geoms[:ind + 1][::-1]

        # We only run the initial coalescence search on the forward run, avoid duplicates 
        self.initialise_run(self.prop_geoms[0],forward)
        if len(self.prop_geoms)>1:
            self.propagate_solution(initSearch=forward)
        print(f"{self.sol} finished, geometries created this walk: ", self.new_geoms)
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
                    # don't overwrite / collide with a distinct solution already there
                    if match_known_solution(hole, wfn, self.dedupEnergyThresh,
                                            self.dedupOverlapThresh, exclude=self.sol):
                        continue
                    wfn.save_to_disk(f"{hole}/{self.sol}")
                anchor = hole   # chain: next hole relaxes from the one we just filled
