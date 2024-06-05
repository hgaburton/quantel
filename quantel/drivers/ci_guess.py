#!/usr/bin/python3

import numpy
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS
from quantel.utils.linalg import random_rot

def ci_guess(ints, config):
    """Generate wavefunctions using standard CI guess as the starting point"""

    print("---------------------------------------------------------------")
    print(" Searching for solutions using configuration interaction guess ")
    print("    + Wavefunction:      {:s}".format(config["wavefunction"]["method"]))
    print("---------------------------------------------------------------")

    print("\n  Generating RHF guess:")
    rhf = RHF(ints)
    rhf.get_orbital_guess()
    DIIS().run(rhf)
    ref_mo = rhf.mo_coeff.copy()
    escf   = rhf.energy
    print(f"    RHF total energy (Eh): {escf: 16.8f}")

    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from quantel.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "casscf":
        from quantel.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "csf":
        errstr = "CI guess is not compatible with CSF wavefunction"
        raise ValueError(errstr)
        
    # Select the optimiser
    optconfig = config["optimiser"][config["optimiser"]["algorithm"]]
    if config["optimiser"]["algorithm"] == "eigenvector_following":
        from quantel.opt.eigenvector_following import EigenFollow as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from quantel.opt.mode_controlling import ModeControl as OPT

    # Get reference MOs and CI vector
    print("\n  Computing initial CI energies (Eh):")
    ref = WFN(ints, **wfnconfig)
    ref.initialise(ref_mo, ref_ci)

    ref_e, ref_ci = numpy.linalg.eigh(ref.ham)
    for ind in config["jobcontrol"]["ci_guess"]:
        print("       Initial state {:4d}: {: 16.8f}".format(ind, ref_e[ind]))  

    # Initialise wavefunction list
    wfn_list = []

    # Perform random search, saving coefficients as we go
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for itest in config["jobcontrol"]["ci_guess"]:
        print("\n  Converging state-specific calculation from initial guess {: 4d}:".format(itest))
        print(  "  --------------------------------------------------------------")
        # Set CI and MO guess according to reference CI problem
        mo_guess = ref_mo.copy()
        ci_guess = ref_ci.copy()
        ci_guess[:,[0,itest]] = ci_guess[:,[itest,0]]

        # Initialise optimisation object
        try: del myfun
        except: pass
        myfun = WFN(ints, **wfnconfig)
        myfun.initialise(mo_guess, ci_guess)

        # Run the optimisation
        myopt = OPT(**optconfig)
        if not myopt.run(myfun, **config["optimiser"]["keywords"]):
            continue

        # Check the Hessian index
        hindices = myfun.get_hessian_index()
        myfun.update_integrals()
        if (hindices[0] != target_index) and (target_index is not None):
            continue
        
        # Compare solution against previously found states
        new = True
        for otherwfn in wfn_list:
            if 1.0 - abs(myfun.overlap(otherwfn)) < config["jobcontrol"]["dist_thresh"]:
                new = False
                break

        # Save the solution if it is a new one!
        if new: 
            if config["wavefunction"]["method"] == "esmf":
                myfun.canonicalize()
            # Get prefix
            count += 1
            tag = "{:04d}".format(count)
            myfun.save_to_disk(tag)
            
            # Deallocate integrals to reduce memory footprint
            myfun.deallocate()
            wfn_list.append(myfun.copy())

    print()
    print(" Search complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("---------------------------------------------------------------")
    print()

    return wfn_list
