from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.uhf import UHF

# Test UHF object with a range of optimisers
print("Test UHF object with a range of optimisers")

for driver in ("libint", "pyscf"):
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
    ints = PySCFIntegrals(mol,xc="PBE0")

    # Initialise UHF object
    wfn = UHF(ints)

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method="gwh")
        LBFGS().run(wfn)
        wfn.print()

        from quantel.opt.diis import DIIS
        wfn.get_orbital_guess(method="gwh")
        DIIS().run(wfn)

        # Test canonicalisation and Hessian eigenvalue
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index(approx_hess=False)
        wfn.print()
        
