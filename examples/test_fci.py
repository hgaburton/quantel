from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCFIntegrals, PySCFMolecule
from quantel.wfn.rhf import RHF
from quantel.wfn.cisolver import FCI
from quantel.opt.diis import DIIS

if __name__ == "__main__":
    # Setup molecule
    mol = PySCFMolecule("mol/h6.xyz", "sto-3g", "angstrom",spin=0,charge=0)
    ints = PySCFIntegrals(mol)

    # Run RHF to get MO coefficients
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    DIIS().run(wfn)

    # Build the integrals
    Ccore = wfn.mo_coeff[:,0:0]
    Cact = wfn.mo_coeff[:,0:ints.nmo()]
    mo_ints = PySCF_MO_Integrals(ints)
    mo_ints.update_orbitals(wfn.mo_coeff,0,ints.nmo())

    # Setup and solve FCI
    ci = FCI(mo_ints, (mol.nalfa(), mol.nbeta()), version=1)
    x, eci = ci.solve(3,verbose=5)
    
    # Verify solution
    if abs(-2.84719213 - x[0]) > 1e-6:
        raise ValueError("FCI energy does not match reference value")
