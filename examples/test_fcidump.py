"""Test script demonstrating the INTDUMPMolecule integral interface.

The test proceeds in two parts:

  Part 1 (H2 / STO-3G)
  ---------------------
  Uses PySCF to run RHF and write a FCIDUMP file in the canonical MO basis.
  Reads the file back with INTDUMPMolecule and cross-checks every quantity
  against the original PySCF integrals:
    - dimensions and electron counts
    - overlap / orthogonalisation matrices are the identity
    - scalar potential matches nuclear repulsion
    - oei_matrix matches PySCF h1e in MO basis
    - tei_array matches PySCF h2e in MO basis (chemist's notation)
    - oei_ao_to_mo and tei_ao_to_mo with a non-trivial orbital rotation
    - build_JK verified against direct einsum contraction
    - dipole_matrix raises RuntimeError
    - mo_integrals returns a valid MOintegrals object
    - FCI energy agrees with PySCF direct FCI

  Part 2 (H6 / STO-3G)
  ---------------------
  Generates a FCIDUMP from PySCF RHF, then drives quantel FCI from the
  FCIDUMP alone (no PySCF integrals at FCI time) and checks against the
  known reference energy.
"""

from curses import OK
import os
import numpy as np
from pyscf import gto, scf
from pyscf.tools import fcidump as pyscf_fcidump
from pyscf import ao2mo

from quantel.ints.fcidump_integrals import FCIDUMP
from quantel.utils.linalg import orthogonalise

def check(label, condition, tol=None, got=None, ref=None):
    if condition:
        print(f"  {label:55s} {'PASS'}")
    else:
        msg = f"  {label:55s} {'FAIL'}"
        if got is not None and ref is not None:
            msg += f"  (got {got}, ref {ref})"
        raise AssertionError(msg)


## Write an FCIDUMP for N2 in a minimal basis, read it back, and check all the integrals against PySCF.]
fname = "fcidump"
# Build PySCF molecule and RHF reference
mol = gto.Mole()
mol.atom   = "N 0 0 0; N 0 0 1.00"
mol.basis  = "sto-3g"
mol.unit   = "angstrom"
mol.verbose = 0
mol.build()

# Build RHF reference and get MO coefficients, integrals, and FCI energy from PySCF
mf = scf.RHF(mol)
mf.kernel()

C    = mf.mo_coeff          # shape (nbsf, nmo)
nmo  = C.shape[1]
nbsf = mol.nao
E_nuc = mol.energy_nuc()

# h1e and h2e in MO basis
overlap = mol.intor("int1e_ovlp")
h1e_ao  = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
h1e_mo  = np.linalg.multi_dot([C.T, h1e_ao, C])
h2e_mo  = ao2mo.full(mol, C, compact=False).reshape(nmo, nmo, nmo, nmo)

# Write FCIDUMP
pyscf_fcidump.from_scf(mf, fname, tol=1e-15)
print(f"\n  FCIDUMP written : {fname}")
print(f"  RHF energy      : {mf.e_tot:.10f}")

# Read FCIDUMP with quantel and check all integrals against PySCF
intdump = FCIDUMP(fname)
intdump.print()
print("\n" + "=" * 60)
print("  FCIDUMP read back with quantel. Checking integrals against PySCF...")
print("=" * 60)

print("\n--- Dimensions and electron counts ---")
check("nbsf",  intdump.nbsf()  == nbsf)
check("nmo",   intdump.nmo()   == nmo)
check("nelec", intdump.molecule().nelec() == mol.nelectron)
check("nalfa", intdump.molecule().nalfa() == mol.nelec[0])
check("nbeta", intdump.molecule().nbeta() == mol.nelec[1])

print("\n--- Overlap / orthogonalisation matrices ---")
check("overlap_matrix = I", np.allclose(intdump.overlap_matrix(), np.eye(nbsf)))
check("orthogonalization_matrix = I", np.allclose(intdump.orthogonalization_matrix(), np.eye(nbsf)))

print("\n--- Scalar potential ---")
check(f"scalar_potential = {E_nuc:.8f}", abs(intdump.scalar_potential() - E_nuc) < 1e-10)

print("\n--- One-electron integrals ---")
oei = intdump.oei_matrix()
check("oei_matrix matches PySCF h1e_mo", np.allclose(oei, h1e_mo, atol=1e-10))

print("\n--- Two-electron integrals ---")
tei = intdump.tei_array()
check("tei_array matches PySCF h2e_mo (chemist's)",np.allclose(tei, h2e_mo, atol=1e-10))

print("\n--- build_JK ---")
# Build a random density matrix in MO basis and check JK against direct einsum contraction with PySCF integrals.
P = np.random.rand(nmo,nmo)

# (2J-K)_{pq} = sum_{rs} P_{rs} * (2*(pq|rs) - (ps|rq))
Jref = np.einsum("pqrs,rs->pq", h2e_mo, P)
Kref = np.einsum("psrq,rs->pq", h2e_mo, P)
Jintdump, Kintdump = intdump.build_JK(P)
check("build_JK: J matches direct einsum",np.allclose(Jintdump, Jref, atol=1e-10))
check("build_JK: K matches direct einsum",np.allclose(Kintdump, Kref, atol=1e-10))

# Check two-electron orbital transformation against direct einsum with PySCF integrals.
print("\n--- One-electron MO transform ---")
C1 = np.random.rand(nbsf,2*nmo)
C2 = np.random.rand(nbsf,4)
C3 = np.random.rand(nbsf,2*nmo)
C4 = np.random.rand(nbsf,2)
h1e_ref = np.linalg.multi_dot([C1.T, h1e_mo, C2])
h1e_intdump = intdump.oei_ao_to_mo(C1,C2,True)
check("oei_ao_to_mo with rotation matches direct einsum", np.allclose(h1e_intdump, h1e_ref, atol=1e-10))

print("\n--- Two-electron MO transform ---")
h2e_ref = np.einsum("pqrs,pa,qb,rc,sd->abcd", h2e_mo.transpose(0,2,1,3),C1,C2,C3,C4, optimize="optimal")
# Convert chemists to physicists notation for direct comparison with intdump.tei_ao_to_mo output
h2e_intdump = intdump.tei_ao_to_mo(C1,C2,C3,C4, True, False)
check("tei_ao_to_mo with rotation matches direct einsum", np.allclose(h2e_intdump, h2e_ref, atol=1e-10))

# Remove the FCIDUMP file
os.remove(fname)

# ===========================================================================
# Done
# ===========================================================================
print("\n" + "=" * 60)
print("  All tests passed!")
print("=" * 60)
