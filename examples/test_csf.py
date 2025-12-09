import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "sto-3g", "angstrom")
    ints = PySCFIntegrals(mol)

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '+-+-')
    wfn.get_orbital_guess(method="gwh")

    # Check gradient and Hessian
    grad = wfn.gradient.copy()
    hess = wfn.hessian.copy()
    grad_check = np.linalg.norm(grad-wfn.get_numerical_gradient())/np.sqrt(grad.size)
    if(grad_check > 1e-5):
        print(f"Gradient check failed. |Analytic - Numerical| = {grad_check: 6.3e}")
    hess_check = np.linalg.norm(hess-wfn.get_numerical_hessian())/np.sqrt(hess.size)
    if(hess_check > 1e-5):
        print(f"Hessian check failed. |Analytic - Numerical| = {hess_check: 6.3e}")

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method=guess)
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index()