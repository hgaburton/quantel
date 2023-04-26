#!/usr/bin/python3

import numpy
from xesla.utils.linalg import random_rot

def random_search(mol, config):
  
    ref_mo = mol.RHF().run().mo_coeff.copy()
    ref_ci = None

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
    print(wfnconfig)

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
        if hindices[0] != config["optimiser"]["keywords"]["index"]: 
            continue
        
        # Compare solution against previously found states
        new = True
        for otherwfn in wfn_list:
            if 1.0 - abs(myfun.overlap(otherwfn)) < config["jobcontrol"]["search"]["dist_thresh"]:
                new = False
                break

        # Save the solution if it is a new one!
        if new: 
            count += 1
            tag = "{:04d}".format(count)
            numpy.savetxt(tag+'.mo_coeff', myfun.mo_coeff, fmt="% 20.16f")
            if myfun.mat_ci is not None:
                numpy.savetxt(tag+'.mat_ci',   myfun.mat_ci, fmt="% 20.16f")
            numpy.savetxt(tag+'.energy',   numpy.array([[myfun.energy, hindices[0], hindices[1], 0.0]]), 
                          fmt="% 18.12f % 5d % 5d % 12.6f")
            
            # Deallocate integrals to reduce memory footprint
            myfun.deallocate()
            wfn_list.append(myfun.copy())

    return wfn_list
