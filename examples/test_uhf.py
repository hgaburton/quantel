import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.uhf import UHF
from quantel.wfn.rhf import RHF

# Test UHF object with a range of optimisers
print("Test UHF object with a range of optimisers")

np.set_printoptions(precision=6, suppress=True,linewidth=10000,edgeitems=5)
for driver in ("libint", "pyscf"):
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "6-31g", "angstrom")
    mol  = PySCFMolecule("h4.xyz", "sto3g", "angstrom")
    ints = PySCFIntegrals(mol,xc="PBE0")

    # Initialise UHF object
    wfn = UHF(ints)
    wfn2 = RHF(ints)

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method="gwh")
        g1 = wfn.gradient.copy()
        g2 = wfn.get_numerical_gradient()
        print(g1)
        print(g2)
        print(np.linalg.norm(g1 - g2))
        g1 = wfn.hessian.copy()
        g2 = wfn.get_numerical_hessian()
        print(g1)
        print(g2)
        print(np.linalg.norm(g1 - g2))
        wfn2.get_orbital_guess(method="gwh")
        g3 = wfn2.hessian.copy()
        print(g3)
        print(g1[:wfn.nrot[0],:wfn.nrot[0]]+g1[wfn.nrot[0]:,wfn.nrot[0]:]+g1[:wfn.nrot[0],wfn.nrot[0]:]+g1[wfn.nrot[0]:,:wfn.nrot[0]])
        quit()
        LBFGS().run(wfn)
        wfn.print()

        from quantel.opt.diis import DIIS
        wfn.get_orbital_guess(method="gwh")
        DIIS().run(wfn)

        # Test canonicalisation and Hessian eigenvalue
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()
        wfn.print()
        
