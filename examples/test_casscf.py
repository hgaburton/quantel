import os
# NOTE: there is some issue with inherent numpy multithreading (not fully diagnosed yet)
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import quantel
from quantel.wfn.ss_casscf import SS_CASSCF
from quantel.opt.eigenvector_following import EigenFollow
from quantel.opt.mode_controlling import ModeControl
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals

# Name of molecule xyz
molxyz = "formaldehyde.xyz"

gradient = []
# Define integrals
for driver in ("libint", "pyscf"):
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    if(driver == "libint"):
        mol  = quantel.Molecule(molxyz, "angstrom")
        ints = quantel.LibintInterface("6-31g", mol) 
    elif(driver == "pyscf"):
        mol  = PySCFMolecule(molxyz, "6-31g", "angstrom")
        ints = PySCFIntegrals(mol)

    wfn = SS_CASSCF(ints,(6,(3,3)))

    mo_guess = np.eye(wfn.nmo)
    ci_guess = np.eye(wfn.ndet)
    wfn.initialise(mo_guess,ci_guess)

    from quantel.opt.lbfgs import LBFGS
    LBFGS(with_transport=False,with_canonical=False).run(wfn)
