#!/usr/bin/python3

import numpy, glob
from pyscf import gto

def follow(ints, config):
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
        #from quantel.opt.test_gmf import GMF as OPT
    elif config["optimiser"]["algorithm"] == "lbfgs":
        from quantel.opt.lbfgs import LBFGS as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from quantel.opt.mode_controlling import ModeControl as OPT
    elif config["optimiser"]["algorithm"] == "adaptive":
        from quantel.opt.lbfgs import LBFGS 
        from quantel.opt.gmf import GMF

    # Initialise wavefunction list
    wfn_list  = []
    name_list = [] 
    e_list    = []
    i_list    = []

    # Reconverge target solutions
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for prefix in config["jobcontrol"]["read_dir"]:
        print(" Reading solutions from directory {:s}".format(prefix))
        #nstates = len(glob.glob(prefix+"*.solution"))
        for old_tag in glob.glob(prefix+"*solution"):
            with open(old_tag, "r") as file: 
                hess_index = file.readline().split()[1]
                hess_index = int(hess_index)
            
            old_tag = old_tag[:-9]
            # Initialise optimisation object
            try: del myfun
            except: pass
            myfun = WFN(ints, **wfnconfig)
            myfun.read_from_disk(old_tag, gcoup=config["jobcontrol"]["gcoup"])
            
            # Run the optimisation
            if config["optimiser"]["algorithm"]=="adaptive": 
                if hess_index==0:
                    lbfgsconfig=optconfig["lbfgs"]
                    myopt = LBFGS(**lbfgsconfig)
                    #surely the different optimisations will need different key words! 
                    if not myopt.run(myfun, **config["optimiser"]["keywords"]):
                        continue
                else: 
                    gmfconfig=optconfig["gmf"]
                    myopt = GMF(**gmfconfig)
                    config["optimiser"]["keywords"]["index"] = hess_index 
                    #then put this into the optimiser
                    #surely the different optimisations will need different key words! 
                    if not myopt.run(myfun, **config["optimiser"]["keywords"]):
                        continue
            else: 
                myopt = OPT(**optconfig)
                if not myopt.run(myfun, **config["optimiser"]["keywords"]):
                    continue
            
            # Check the Hessian index
            myfun.canonicalize()
            if config["jobcontrol"]["nohess"]:
                myfun.hess_index = (0,0,0)     
                hindices = myfun.hess_index   
            else:
                myfun.get_davidson_hessian_index()
                hindices = myfun.hess_index
                # Yes need to set target index = None to prevent skipping the rest of the loop
                if (hindices[0] != target_index) and (target_index is not None):
                    continue

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
                # Name it according to the followed solution
                count += 1
                tag = old_tag[-4:]

                # Save the object to disk - only if want to 
                if config["jobcontrol"]["save_solns"]: 
                    myfun.save_to_disk(tag)

                # Save energy and indices
                e_list.append(myfun.energy)
                i_list.append(hindices[0])

                # Deallocate integrals to reduce memory footprint
                myfun.deallocate()
                wfn_list.append(myfun.copy())
                name_list.append(old_tag[2:]) 
            else: 
                print("  Solution matches previous solution...",prev+1)

        # Print a new line
        print()

    numpy.savetxt('energy_list', numpy.array([e_list]),fmt="% 16.10f")
    numpy.savetxt('ind_list', numpy.array([i_list]),fmt="% 5d")
    numpy.savetxt('name_list', numpy.array([name_list]), fmt='%s')

    print()
    print(" Read from file complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("--------------------------------------------------------------")
    print()

    return wfn_list
