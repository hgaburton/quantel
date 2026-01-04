from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF

if __name__ == "__main__":
    print("===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
       # Setup molecule and integrals
    mol  = PySCFMolecule("mol/formaldehyde.xyz", "def2svp", "angstrom")
    ints = PySCFIntegrals(mol,xc='pbe0')

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '+-')
    wfn.get_orbital_guess(method="gwh")

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
        wfn.get_davidson_hessian_index(approx_hess=False)