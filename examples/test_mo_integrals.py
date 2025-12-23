"""
An illustration of how to use PySCF MO integral objects
"""
import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFIntegrals, PySCFMolecule, PySCF_MO_Integrals
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS

if __name__ == "__main__":
    # Initialise H6 molecule
    mol = PySCFMolecule("mol/h6.xyz", "sto-3g", "angstrom",spin=0,charge=0)
    ints = PySCFIntegrals(mol)
    # Initialise RHF object from integrals
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    
    # Run DIIS to convergence
    DIIS().run(wfn)
    
    # Perform MO transfrom
    C = wfn.mo_coeff.copy()
    
    # Initialise MO integrals object
    mo_ints = PySCF_MO_Integrals(ints)

    # Compute integrals for alpha oei
    h1e = mo_ints.compute_oei(C,True)

    # Compute eri for alpha-alpha tei (these are antisymmetrized)
    h2e_aa = mo_ints.compute_tei(C,True,True)

    # Compute eri for alpha-beta tei (these are not antisymmetrized)
    h2e_ab = mo_ints.compute_tei(C,True,False)

    # Can also compute using JK builds (direct implementation)
    h2e_aa_jk = mo_ints.compute_tei_from_JK(C,antisym=True)
    h2e_ab_jk = mo_ints.compute_tei_from_JK(C,antisym=False)

    # Verify that both implementations give the same result
    assert np.allclose(h2e_aa,h2e_aa_jk)
    assert np.allclose(h2e_ab,h2e_ab_jk)

    print("  MO integral computations successful!")
