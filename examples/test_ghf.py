import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.ghf import GHF
from quantel.opt.lbfgs import LBFGS
from quantel.opt.diis import DIIS

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing GHF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("h3.xyz", "3-21g", "angstrom",spin=1,charge=0)
    ints = PySCFIntegrals(mol)

    # Initialise GHF object
    wfn = GHF(ints)
    # Set initial coefficients from identity
    wfn.initialise(np.eye(wfn.nmo,wfn.nmo))

    # Check gradient and Hessian
    grad = wfn.gradient.copy()
    hess = wfn.hessian.copy()
    grad_check = np.linalg.norm(grad-wfn.get_numerical_gradient())/np.sqrt(grad.size)
    if(grad_check > 1e-5):
        print(f"Gradient check failed. |Analytic - Numerical| = {grad_check: 6.3e}")
    hess_check = np.linalg.norm(hess-wfn.get_numerical_hessian())/np.sqrt(hess.size)
    if(hess_check > 1e-5):
        print(f"Hessian check failed. |Analytic - Numerical| = {hess_check: 6.3e}")

    # Test LBFGS
    LBFGS().run(wfn)

    # Test DIIS
    wfn.initialise(np.eye(wfn.nmo,wfn.nmo))
    DIIS().run(wfn)

    # Test canonicalisation and Hessian eigenvalue
    wfn.canonicalize()
    # Test Hessian index
    wfn.get_davidson_hessian_index()
