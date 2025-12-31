import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF

np.set_printoptions(linewidth=10000,precision=6,suppress=True)

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    #mol  = PySCFMolecule("mol/formaldehyde.xyz", "sto-3g", "angstrom")
    mol  = PySCFMolecule("mol/h6.xyz", "sto-3g", "angstrom")
    ints = PySCFIntegrals(mol)

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '++--')
    wfn.get_orbital_guess(method="gwh")

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method=guess)
        np.random.seed(8)
        vec = np.random.rand(wfn.dim)
        hv1 = wfn.hessian @ vec
        hv2 = wfn.hess_on_vec(vec)
        hv3 = wfn.get_numerical_hessian() @ vec
        hv1mat = np.zeros((wfn.nmo,wfn.nmo))
        hv2mat = np.zeros((wfn.nmo,wfn.nmo))
        hv3mat = np.zeros((wfn.nmo,wfn.nmo))
        hv1mat[wfn.rot_idx]=hv1
        hv2mat[wfn.rot_idx]=hv2
        hv3mat[wfn.rot_idx]=hv3
        print("hessian @ vec")
        print(hv1mat)
        print("hess_on_vec")
        print(hv2mat)
        print("num_hess_on_vec")
        print(hv3mat)
        print(np.linalg.norm(hv1-hv2))
        print(np.linalg.norm(hv2-hv3))
        quit()

        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()
