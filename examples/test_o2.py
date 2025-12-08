import quantel
import numpy as np
from scipy.linalg import eigh
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals, PySCF_MO_Integrals
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import CustomCI

molxyz = "o2.xyz"
basis = 'aug-cc-pvqz'

# Test RHF object with a range of optimisers
print("Test CSF object with a range of optimisers")

for driver in ["pyscf"]:
    print("\n===============================================")
    print(f" Testing '{driver}' integral method")
    print("===============================================")
    mol  = PySCFMolecule(molxyz, basis, "angstrom")
    ints = PySCFIntegrals(mol,xc=',tpss')

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '++')

    # Setup optimiser
    from quantel.opt.lbfgs import LBFGS
    wfn.get_orbital_guess(method="rohf")
    LBFGS().run(wfn)
    # Test canonicalisation 
    wfn.canonicalize()
    F = wfn.get_generalised_fock()[0][:wfn.nocc,:wfn.nocc]
    d = wfn.get_rdm12(only_occ=True)[0]
    print(F.shape)
    print(d.shape)
    e,v = eigh(-F,d)
    for ei in e:
        print(f"{ei*27.2114: 16.8f}")

    # Initialise CSF object for an open-shell singlet state
    wfn = CSF(ints, '+-')
    wfn.get_orbital_guess(method="rohf")
    LBFGS().run(wfn)

    # Initialise CSF object for an open-shell singlet state
    mol  = PySCFMolecule('o2.xyz',basis, "angstrom",charge=1,spin=1)
    ints = PySCFIntegrals(mol,xc=',tpss')
    wfn = CSF(ints, '+')
    wfn.get_orbital_guess(method="rohf")
    LBFGS().run(wfn)
    
    # Initialise CSF object for an open-shell singlet state
    mol  = PySCFMolecule('o2p.xyz',basis, "angstrom",charge=1,spin=1)
    ints = PySCFIntegrals(mol,xc=',tpss')
    wfn = CSF(ints, '+')
    wfn.get_orbital_guess(method="rohf")
    LBFGS().run(wfn)
