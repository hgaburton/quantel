#!/usr/bin/python3

import numpy, glob
from pyscf import gto
from quantel.utils.linalg import random_rot

def rand_fromfile(ints, config):
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

    # Initialise wavefunction list
    wfn_list  = []
    e_list    = []
    i_list    = []

    # Reconverge target solutions
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    # this is counting for all the input directories
    for prefix in config["jobcontrol"]["read_dir"]:
        print(" Reading Solutions from directory {:s}".format(prefix))
        #Need to count the number of states to converge
        nstates = len(glob.glob(prefix+"*.solution"))
        for i in range(nstates):
            old_tag = "{:s}{:04d}".format(prefix, i+1)
            mo_range = config["jobcontrol"]["search"]["mo_rot_range"]
            # Perform random rotation on these guess states
            for itest in range(config["jobcontrol"]["search"]["nsample"]):
                # Initialise optimisation object
                try: del myfun
                except: pass
                myfun = WFN(ints, **wfnconfig)
                myfun.read_from_disk(old_tag, gcoup=config["jobcontrol"]["gcoup"])
            
                #
                mo_guess = myfun.mo_coeff.dot(random_rot(myfun.nmo, -mo_range, mo_range)) #so the random rotations are performed from the ROHF reference, for the given random seed?
                myfun.mo_coeff = mo_guess
                myfun.update()
                # Run the optimisation
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
                    print("  Found new solution!")
                else: 
                    print("  Solution matches previous solution...",prev+1)
            

            # We also want code targeting higher index states: 
            # best to maybe look at the H2 example 
            #this should also leave a comment in the file for the solution 
            #_ ,x = myfun.get_davidson_hessian_index(approx_hess=False)
            #curr_index = myfun.hess_index
            #lowest_positive_index = np.sum(curr_index) 
            # do we do a davidson explicitly here in order to extract the number of
            #search_direction = x[:,lowest_positive_index]

            # then do a line search in this direction to find a maximum in the direction - so since we know this direction is positve hessian eigenvalue - 
        # Print a new line
        print()

    numpy.savetxt('energy_list', numpy.array([e_list]),fmt="% 16.10f")
    numpy.savetxt('ind_list', numpy.array([i_list]),fmt="% 5d")

    print()
    print(" Read from file complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("--------------------------------------------------------------")
    print()

    return wfn_list
