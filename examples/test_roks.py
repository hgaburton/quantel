import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals, PySCF_MO_Integrals
from quantel.wfn.roks import ROKS
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import CustomCI
from quantel.opt.gmf import GMF
from quantel.opt.lbfgs import LBFGS

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "def2svp", "angstrom")
    ints = PySCFIntegrals(mol,xc='pbe0')
    print(ints)

    # Initialise CSF object for an open-shell singlet state
    wfn = ROKS(ints, '+-')
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
        wfn.get_davidson_hessian_index()