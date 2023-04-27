#!/usr/bin/python3

import numpy
from xesla.utils.linalg import random_rot

def random_search(mol, config):
    """Perform a random search for multiple solutions"""

    # Get reference RHF wavefunction
    ref_mo = mol.RHF().run().mo_coeff.copy()
    ref_ci = None

    # Get information about the wavefunction defintion
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from xesla.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "casscf":
        from xesla.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "csf":
        from xesla.wfn.csf import CSF as WFN
        ndet = 0
        
    # Get variables
    nmo  = ref_mo.shape[1]

    # Select the optimiser
    optconfig = config["optimiser"][config["optimiser"]["algorithm"]]
    if config["optimiser"]["algorithm"] == "eigenvector_following":
        from xesla.opt.eigenvector_following import EigenFollow as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from xesla.opt.mode_controlling import ModeControl as OPT

    # Set numpy random seed
    numpy.random.seed(config["jobcontrol"]["search"]["seed"])

    # Initialise wavefunction list
    wfn_list = []

    # Perform random search, saving coefficients as we go
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for itest in range(config["jobcontrol"]["search"]["nsample"]):
        # Randomly perturb CI and MO coefficients
        mo_guess = ref_mo.dot(random_rot(nmo,  -numpy.pi, numpy.pi))
        if(ref_ci) is None:
            ci_guess = None
        else:
            ci_guess = ref_ci.dot(random_rot(ndet, -numpy.pi, numpy.pi))

        # Initialise optimisation object
        try: del myfun
        except: pass
        myfun = WFN(mol, **wfnconfig)
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
            # Get the prefix for this solution
            count += 1
            tag = "{:04d}".format(count)

            # Save the object to disck
            myfun.save_to_disk(tag)

            # Append wavefunction to our list
            wfn_list.append(myfun.copy())

    return wfn_list
