import time
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
    mol  = PySCFMolecule("mol/h6.xyz", "6-31g", "angstrom")
    ints = PySCFIntegrals(mol)

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '+-')
    wfn.get_orbital_guess(method="gwh")

    # Setup optimiser
    T1 = []
    T2 = []
    T3 = []
    for repeat in range(10):
        for guess in ("gwh", "core"):
            print("\n************************************************")
            print(f" Testing '{guess}' initial guess method")
            print("************************************************")
            from quantel.opt.lbfgs import LBFGS
            wfn.get_orbital_guess(method=guess)
            np.random.seed(8)
            vec = np.random.rand(wfn.dim)
            # Compare time for hv1 and hv2 computation
            start = time.time()
            hv1 = wfn.hessian @ vec
            end = time.time()
            print(ints.nbsf())
            print(f"Hessian @ vec time: {1000*(end-start):.6f} ms")
            T1.append(1000*(end-start))
            start = time.time()
            hv2 = wfn.hess_on_vec(vec)
            end = time.time()
            T2.append(1000*(end-start))
            print(f"  Hess_on_vec time: {1000*(end-start):.6f} ms")
            start = time.time()
            hv3 = wfn.approx_hess_on_vec(vec)
            end = time.time()
            T3.append(1000*(end-start))
            print(f"  approx_hess_on_vec time: {1000*(end-start):.6f} ms")
            hv1mat = np.zeros((wfn.nmo,wfn.nmo))
            hv2mat = np.zeros((wfn.nmo,wfn.nmo))
            hv3mat = np.zeros((wfn.nmo,wfn.nmo))
            hv1mat[wfn.rot_idx]=hv1
            hv2mat[wfn.rot_idx]=hv2
            hv3mat[wfn.rot_idx]=hv3
            #print("hessian @ vec")
            #print(hv1mat)
            #print("hess_on_vec")
            #print(hv2mat)
            print(np.linalg.norm(hv1-hv2))
            print(np.linalg.norm(hv1-hv3))
            print(hv1mat)
            print(hv2mat)
            print(hv3mat)
            quit()
    print(ints.nbsf(),np.mean(T1), np.mean(T2))
    quit()

    #    LBFGS().run(wfn)
    #    
    #    # Test canonicalisation 
    #    wfn.canonicalize()
    #    # Test Hessian index
    #    wfn.get_davidson_hessian_index()
