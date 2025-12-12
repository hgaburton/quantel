import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
import datetime

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "aug-cc-pvdz", "angstrom")
    ints = PySCFIntegrals(mol)
    df_ints = PySCFIntegrals(mol,with_df=True)

    # Initialise CSF object for an open-shell singlet state
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    dm = wfn.dens
    # Time the JK builds with and without density fitting
    t0 = datetime.datetime.now()
    JK = ints.build_JK(dm)
    t1 = datetime.datetime.now()
    print("Conventional JK build time: ",(t1-t0).total_seconds())
    print(JK)
    t0 = datetime.datetime.now()
    JK = df_ints.build_JK(dm)
    t1 = datetime.datetime.now()
    print("Density-fitted JK build time: ",(t1-t0).total_seconds())
    print(JK)

    vJ = ints.build_multiple_JK([dm,dm],[dm,dm],hermi=1)
    vJ_df = df_ints.build_multiple_J([dm,dm],hermi=1)
    quit()

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
