#!/usr/bin/python3

import numpy, glob
from pyscf import gto

def ladder_from_file(ints, config):
    """Read wavefunctions from solutions that are saved to file"""

    print("-----------------------------------------------")
    print(" Reading solutions from file")
    print("    + Wavefunction:       {:s}".format(config["wavefunction"]["method"]))
    print("-----------------------------------------------")

    # Get information about the wavefunction
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from quantel.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "casscf":
        from quantel.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "csf":
        from quantel.wfn.csf import CSF as WFN
    elif config["wavefunction"]["method"] == "rhf":
        from quantel.wfn.rhf import RHF as WFN
    elif config["wavefunction"]["method"] == "roks":
        from quantel.wfn.roks import ROKS as WFN
    else:
        raise ValueError("Wavefunction method not recognised")

    # Select the optimiser
    optconfig = config["optimiser"][config["optimiser"]["algorithm"]]
    if config["optimiser"]["algorithm"] == "eigenvector_following":
        from quantel.opt.eigenvector_following import EigenFollow as OPT
    elif config["optimiser"]["algorithm"] == "lsr1":
        from quantel.opt.lsr1 import SR1 as OPT
    elif config["optimiser"]["algorithm"] == "gmf":
        from quantel.opt.gmf import GMF as OPT
    elif config["optimiser"]["algorithm"] == "lbfgs":
        from quantel.opt.lbfgs import LBFGS as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from quantel.opt.mode_controlling import ModeControl as OPT
    elif config["optimiser"]["algorithm"] == "diis":
        from quantel.opt.diis import DIIS as OPT
    elif config["optimiser"]["algorithm"] == "hybridef":
        from quantel.opt.hybrid_ef import HybridEF as OPT

    # Initialise wavefunction list
    wfn_list  = []
    e_list    = []
    i_list    = []

    # Read in solutions, their indices and then try and do the ladder forward and backwards?     
    max_index = config["optimiser"]["keywords"]["index"]
    # Delete the index dictionary element so we can fix it ourselves later 
    del config["optimiser"]["keywords"]["index"] 
    
    count = 0
    for prefix in config["jobcontrol"]["read_dir"]:
        print(" Reading solutions from directory {:s}".format(prefix))
        # Need to count the number of states to converge
        nstates = len(glob.glob(prefix+"*.solution"))
        #for i in range(nstates):
        for i in glob.glob(prefix + "*.solution"):
            #old_tag = "{:s}{:04d}".format(prefix, i+1)
            old_tag = i[:-9]
            
            with open(i, "r") as f: 
                hess_index = f.readline().split()[1]
                hess_index = int(hess_index) 
            
            # Initialise wave function object
            try: del myfun
            except: pass
            myfun = WFN(ints, **wfnconfig)
            myfun.read_from_disk(old_tag, gcoup=config["jobcontrol"]["gcoup"])
            # Initialise optimiser object
            myopt = OPT(**optconfig)
            down_inds = [ a for a in range(hess_index-1, -1, -1)] 
            up_inds = [ a for a in range(hess_index+1, max_index) ] 
            total_inds = down_inds + up_inds
            
            Cref = myfun.mo_coeff.copy()    
            for tindex in total_inds:
                # initialise optimisation
                if not myopt.run(myfun, **config["optimiser"]["keywords"], index=tindex):
                    continue

                # Check the Hessian index
                myfun.canonicalize()
                if config["jobcontrol"]["nohess"]:
                    myfun.hess_index = (0,0,0)     
                    hindices = myfun.hess_index   
                else:
                    myfun.get_davidson_hessian_index()
                    hindices = myfun.hess_index

                # Compare solution against previously found states
                new = True
                for prev, otherwfn in enumerate(wfn_list):
                    if abs(myfun.energy - otherwfn.energy) < config["jobcontrol"]["dist_thresh"]:
                      if 1.0 - abs(myfun.overlap(otherwfn)) < config["jobcontrol"]["dist_thresh"]:
                        new = False
                        break

                # Save the solution if it is a new one!
                if new: 
                    if config["wavefunction"]["method"] == "esmf":
                        myfun.canonicalize()
                    # Get the prefix for this solution
                    count += 1
                    tag = "{:04d}".format(count)

                    # Save the object to disck
                    myfun.save_to_disk(tag)

                    # Save energy and indices
                    e_list.append(myfun.energy)
                    i_list.append(hindices[0])

                    # Deallocate integrals to reduce memory footprint
                    myfun.deallocate()
                    wfn_list.append(myfun.copy())
                else: 
                    print("  Solution matches previous solution...",prev+1)

                if tindex==0: 
                    #Reset the wfn object
                    myfun.mo_coeff = Cref 
                    myfun.update() 

        # Print a new line
        print()

    numpy.savetxt('energy_list', numpy.array([e_list]),fmt="% 16.10f")
    numpy.savetxt('ind_list', numpy.array([i_list]),fmt="% 5d")

    print()
    print(" Read from file complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("--------------------------------------------------------------")
    print()

    return wfn_list
