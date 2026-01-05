import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF

molxyz = "formaldehyde.xyz"

# Test CSF object with a range of optimisers
print("Test CSF object with a range of optimisers")

for driver in ["pyscf"]:
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    if(driver == "libint"):
        mol  = quantel.Molecule(molxyz, "angstrom")
        ints = quantel.LibintInterface("6-31g", mol) 
    elif(driver == "pyscf"):
        mol  = PySCFMolecule(molxyz, "aug-ccpvtz", "angstrom")
        ints = PySCFIntegrals(mol)

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '+-')

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method="gwh")
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()




