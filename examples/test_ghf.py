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
    mol  = PySCFMolecule("mol/h3.xyz", "sto3g", "angstrom",spin=1,charge=0)
    ints = PySCFIntegrals(mol)
    nmo  = 2*ints.nmo()
    Cinit = np.eye(nmo,nmo)

    # Initialise GHF object
    wfn = GHF(ints)
    # Set initial coefficients from identity
    wfn.initialise(Cinit)
    # Test LBFGS
    LBFGS().run(wfn)

    # Test canonicalisation and Hessian eigenvalue
    wfn.canonicalize()
    # Test Hessian index
    wfn.get_davidson_hessian_index(approx_hess=False)
    
    # Test DIIS
    wfn.initialise(Cinit)
    DIIS().run(wfn)

