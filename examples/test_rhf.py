import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF

np.set_printoptions(linewidth=10000,precision=6,suppress=True,edgeitems=50)
if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing RHF object with a range of optimisers")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("h4.xyz", "sto3g", "angstrom")
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
        
        # Test canonicalisation and Hessian eigenvalue
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()

        from quantel.opt.diis import DIIS
        wfn.get_orbital_guess(method="gwh")
        DIIS().run(wfn)
        
        # Test canonicalisation and Hessian eigenvalue
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()
