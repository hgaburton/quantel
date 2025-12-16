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

    # Test LBFGS
    LBFGS().run(wfn)

    # Test DIIS
    wfn.initialise(np.eye(wfn.nmo,wfn.nmo))
    DIIS().run(wfn)

    # Test canonicalisation and Hessian eigenvalue
    wfn.canonicalize()
    # Test Hessian index
    wfn.get_davidson_hessian_index()
