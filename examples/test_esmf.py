import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFIntegrals, PySCFMolecule
from quantel.opt.lbfgs import LBFGS
from quantel.wfn.rhf import RHF
from quantel.wfn.esmf import ESMF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing RHF object with a range of optimisers")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("mol/formaldehyde.xyz", "sto3g", "angstrom")
    ints = PySCFIntegrals(mol)

    # Initialise RHF object
    wfn = ESMF(ints)

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        np.random.seed(7)
        wfn.initialise(np.random.rand(wfn.nbsf,wfn.nmo), np.random.rand(wfn.ndet,wfn.ndet))
        wfn.canonicalize()
        LBFGS(with_canonical=False).run(wfn,maxit=10000)
