import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
from pygnme import utils
import time

# Test RHF object with a range of optimisers
print("Test RHF object with a range of optimisers")

for driver in ("libint", "pyscf"):
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    if(driver == "libint"):
        mol  = quantel.Molecule("formaldehyde.xyz", "angstrom")
        ints = quantel.LibintInterface("6-31g", mol) 
    elif(driver == "pyscf"):
        mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
        ints = PySCFIntegrals(mol)

    # Initialise RHF object
    wfn = RHF(ints)

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method="gwh")
        LBFGS().run(wfn)

        from quantel.opt.diis import DIIS
        wfn.get_orbital_guess(method="gwh")
        DIIS().run(wfn)
