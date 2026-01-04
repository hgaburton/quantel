import time
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF
from quantel.wfn.roks import ROKS
from quantel.wfn.uhf import UHF

np.set_printoptions(linewidth=10000,precision=6,suppress=True)

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    #mol  = PySCFMolecule("mol/formaldehyde.xyz", "sto-3g", "angstrom")
    mol  = PySCFMolecule("mol/h4.xyz", "sto3g", "angstrom",spin=0)
    ints = PySCFIntegrals(mol,xc='pbe0')

    # Initialise CSF object for an open-shell singlet state
    wfn = ROKS(ints, '+-')
    wfn.get_orbital_guess(method="gwh")
    wfn2 = UHF(ints)


    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method=guess)
        #LBFGS().run(wfn)

        np.random.seed(8)
        vec = np.random.rand(wfn.dim)
        # Test gradient
        grad = wfn.gradient
        grad_num = wfn.get_numerical_gradient(eps=1e-3)
        print(grad)
        print(grad_num)
        hess = wfn.hessian
        print(hess)
        print("Analytic Hessian diagonal:")
        print(np.abs(np.diag(hess)))
        print("Preconditioner:")
        print(wfn.get_preconditioner(abs=True,include_fxc=True))
        hess_num = wfn.get_numerical_hessian()
        print(hess_num)
        print(np.linalg.norm(hess - hess_num))
        
        # Compare time for hv1 and hv2 computation
        hv2 = wfn.hess_on_vec(vec)
        hv2mat = np.zeros((wfn.nmo,wfn.nmo))
        hv2mat[wfn.rot_idx]=hv2
        print("hess_on_vec")
        print(hv2mat)

        hv1 = wfn.hessian @ vec
        hv1mat = np.zeros((wfn.nmo,wfn.nmo))
        hv1mat[wfn.rot_idx]=hv1
        print("hessian @ vec")
        print(hv1mat)
        print(f"          Analytic: {np.linalg.norm(hv1-hv2): 8.5e}")
        quit()
        hv3 = wfn.approx_hess_on_vec(vec)
        hv3mat = np.zeros((wfn.nmo,wfn.nmo))
        hv3mat[wfn.rot_idx]=hv3
        print("approx_hess_on_vec")
        print(hv3mat)
        print(f"Approx_hess_on_vec: {np.linalg.norm(hv2-hv3): 8.5e}")


        hv4 = wfn.get_numerical_hessian() @ vec
        hv4mat = np.zeros((wfn.nmo,wfn.nmo))
        hv4mat[wfn.rot_idx]=hv4
        print("num_hess_on_vec")
        print(hv4mat)
        print(f" Numerical Hessian: {np.linalg.norm(hv2-hv4): 8.5e}")

        quit()
    quit()

    #    LBFGS().run(wfn)
    #    
    #    # Test canonicalisation 
    #    wfn.canonicalize()
    #    # Test Hessian index
    #    wfn.get_davidson_hessian_index()
