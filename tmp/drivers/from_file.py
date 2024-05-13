#!/usr/bin/python3

import numpy, glob

def from_file(mol, config):
    """Read wavefunctions from solutions that are saved to file"""

    print("-----------------------------------------------")
    print(" Reading solutions from file")
    print("    + Wavefunction:       {:s}".format(config["wavefunction"]["method"]))
    print("-----------------------------------------------")

    # Get information about the wavefunction
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from exelsis.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "casscf":
        from exelsis.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "csf":
        from exelsis.wfn.csf import CSF as WFN
        ndet = 0

    # Select the optimiser
    optconfig = config["optimiser"][config["optimiser"]["algorithm"]]
    if config["optimiser"]["algorithm"] == "eigenvector_following":
        from exelsis.opt.eigenvector_following import EigenFollow as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from exelsis.opt.mode_controlling import ModeControl as OPT

    # Initialise wavefunction list
    wfn_list  = []
    e_list    = []
    i_list    = []
    ept2_list = []

    # Reconverge target solutions
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for prefix in config["jobcontrol"]["read_dir"]:
        print(" Reading solutions from directory {:s}".format(prefix))
        # Need to count the number of states to converge
        nstates = len(glob.glob(prefix+"*.mo_coeff"))
        for i in range(nstates):
            old_tag = "{:s}{:04d}".format(prefix, i+1)

            # Initialise optimisation object
            try: del myfun
            except: pass
            myfun = WFN(mol, **wfnconfig)
            myfun.read_from_disk(old_tag)

            # Run the optimisation
            myopt = OPT(**optconfig)
            if not myopt.run(myfun, **config["optimiser"]["keywords"]):
                continue

            # Get the Hessian index
            hindices = myfun.get_hessian_index()
            if (hindices[0] != target_index) and (target_index is not None):
                continue

            # Compare solution against previously found states
            new = True
            for prev, otherwfn in enumerate(wfn_list):
                if 1.0 - abs(myfun.overlap(otherwfn)) < config["jobcontrol"]["dist_thresh"]:
                    new = False
                    break

            # Save the solution if it is a new one!
            if new: 
                if config["wavefunction"]["method"] == "esmf":
                    myfun.canonicalise()
                # Get the prefix for this solution
                count += 1
                tag = "{:04d}".format(count)

                # Save the object to disck
                myfun.save_to_disk(tag)

                # Save energy and indices
                e_list.append(myfun.energy)
                i_list.append(hindices[0])
                if(config["jobcontrol"]["nevpt2"]): 
                    if config["wavefunction"]["method"] != "casscf":
                        raise ValueError("NEVPT2 is only compatible with CASSCF function")
                    ept2_list.append(myfun.get_pt2_correction())



                # Deallocate integrals to reduce memory footprint
                myfun.deallocate()
                wfn_list.append(myfun.copy())
            else: 
                print("  Solution matches previous solution...",prev+1)

        # Print a new line
        print()

    numpy.savetxt('energy_list', numpy.array([e_list]),fmt="% 16.10f")
    numpy.savetxt('ind_list', numpy.array([i_list]),fmt="% 5d")
    if(config["jobcontrol"]["nevpt2"]): 
        numpy.savetxt('energy_pt2_list', numpy.array([ept2_list]),fmt="% 16.10f")

    print()
    print(" Read from file complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("--------------------------------------------------------------")
    print()

    return wfn_list
