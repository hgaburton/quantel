#!/usr/bin/python3

import numpy
from scipy.linalg import expm 
from quantel.utils.linalg import random_rot
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS
from quantel import LibintInterface
from quantel.opt.lbfgs import LBFGS
from quantel.wfn.csf import GenealogicalCSF as CSF

def random_search(ints, config):
    """Perform a random search for multiple solutions"""

    print("-----------------------------------------------")
    print(" Searching for solutions using random searches ")
    print("    + Wavefunction:      {:s}".format(config["wavefunction"]["method"]))
    print("    + Number of samples: {:d}".format(config["jobcontrol"]["search"]["nsample"]))
    print("-----------------------------------------------")

    # Get RHF orbitals
    ms = ints.molecule().nalfa() - ints.molecule().nbeta()
    if(ms==0):
        print("\nRunning initial closed-shell RHF calculation...")
        mf = RHF(ints)
        mf.get_orbital_guess()
        DIIS().run(mf,thresh=1e-7)
        ref_mo = mf.mo_coeff.copy()
    else:
        print(f"\nRunning initial high-spin ROHF calculation with multiplicity {ms+1: 3d}...")
        mf = CSF(ints, '+'*ms)
        mf.get_orbital_guess()
        LBFGS().run(mf,maxit=500)
        ref_mo = mf.mo_coeff.copy()
    ref_ci = None

    # Get information about the wavefunction defintion
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from quantel.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "casscf":
        from quantel.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "csf":
        from quantel.wfn.csf import GenealogicalCSF as WFN
    elif config["wavefunction"]["method"] == "rhf":
        from quantel.wfn.rhf import RHF as WFN
    else:
        raise ValueError("Wavefunction method not recognised")
        
    # Get variables
    nmo  = ref_mo.shape[1]

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

    # Set numpy random seed
    numpy.random.seed(config["jobcontrol"]["search"]["seed"])

    # Initialise wavefunction list
    wfn_list = []

    # Perform random search, saving coefficients as we go
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for itest in range(config["jobcontrol"]["search"]["nsample"]):
        # Randomly perturb CI and MO coefficients
        mo_range = config["jobcontrol"]["search"]["mo_rot_range"]
        mo_guess = ref_mo.dot(random_rot(nmo,  -mo_range, mo_range))
        if(ref_ci) is None:
            ci_guess = None
        else:
            ci_guess = ref_ci.dot(random_rot(ndet, -numpy.pi, numpy.pi))

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
        myfun.canonicalize()
        myfun.get_davidson_hessian_index()
        hindices = myfun.hess_index
        if (hindices[0] != target_index) and (target_index is not None):
            continue
        
        # Compare solution against previously found states
        new = True
        for otherwfn in wfn_list:
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

            # Append wavefunction to our list
            wfn_list.append(myfun.copy())

    print()
    print(" Search complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("------------------------------------------------------")
    print()

    return wfn_list
