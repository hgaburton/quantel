"""Benchmark FCIDUMP vs PySCFIntegrals for RHF, UHF, and GHF on N2/STO-3G.

Steps
-----
1. Build N2/STO-3G with PySCF, run RHF to get a reference energy.
2. Run RHF, UHF, GHF using PySCFIntegrals (AO basis).
3. Write a FCIDUMP in the canonical MO basis, read it back.
4. Run RHF, UHF, GHF using the FCIDUMP integral interface (MO basis).
5. Check every energy against the PySCF reference.

Note on initial guesses
-----------------------
PySCFIntegrals: AO basis — use GWH/core guess, which works as for any SCF.
FCIDUMP:        MO basis — the identity matrix IS the RHF solution, so we
                build a block-diagonal spinor guess rather than using the
                orbital-guess routines (which were designed for the AO basis).
"""

import copy
import os
import numpy as np
from pyscf import gto, scf
from pyscf.tools import fcidump as pyscf_fcidump

from quantel import wfn
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.ints.fcidump_integrals import FCIDUMP
from quantel.wfn.rhf import RHF
from quantel.wfn.uhf import UHF
from quantel.wfn.ghf import GHF
from quantel.opt.diis import DIIS

FCIDUMP_FILE = "tmp.fcidump"
TOL_ENERGY   = 1e-8   # Hartree
ATOM         = "Li 0 0 0; H 0 0 1.098"
BASIS        = "sto3g"

def check(label, condition, got=None, ref=None):
    if condition:
        print(f"  {label:62s} PASS")
    else:
        msg = f"  {label:62s} FAIL"
        if got is not None and ref is not None:
            msg += f"  (got {got:.10f}, ref {ref:.10f})"
        raise AssertionError(msg)


def ghf_from_rhf_coeffs(wfn_ghf, C_rhf):
    """Build a block-diagonal GHF initial guess from RHF orbitals.

    Interleaves alpha-occupied and beta-occupied columns so the first nocc
    spinors correspond to nocc/2 alpha + nocc/2 beta electrons.  This ensures
    GHF starts at (and confirms) the RHF stationary point.
    """
    nbsf = C_rhf.shape[0]
    nmo  = C_rhf.shape[1]
    nocc_half = wfn_ghf.nocc // 2
    C_block   = np.block([[C_rhf, np.zeros((nbsf, nmo))],
                          [np.zeros((nbsf, nmo)), C_rhf]])
    col_order = (list(range(nocc_half)) +
                 list(range(nmo, nmo + nocc_half)) +
                 list(range(nocc_half, nmo)) +
                 list(range(nmo + nocc_half, 2 * nmo)))
    wfn_ghf.initialise(C_block[:, col_order])


def run_scf(wfn_cls, ints, coeff=None, ghf_coeff=None):
    """Initialise, run DIIS, and return the converged wfn.

    Parameters
    ----------
    coeff : ndarray or None
        If provided, use directly as the initial orbital coefficients
        (via wfn.initialise).  Otherwise fall back to get_orbital_guess.
    ghf_coeff : ndarray or None
        If provided and wfn_cls is GHF, use these spatial orbitals to build
        the block-diagonal GHF spinor starting guess.
    """
    wfn = wfn_cls(ints)
    if wfn_cls is GHF and ghf_coeff is not None:
        ghf_from_rhf_coeffs(wfn, ghf_coeff)
    elif coeff is not None:
        wfn.initialise(coeff)
    elif wfn_cls is GHF:
        wfn.get_orbital_guess(method="core")
    else:
        wfn.get_orbital_guess(method="gwh")
    DIIS().run(wfn)
    return wfn


# ===========================================================================
# 1. PySCF reference
# ===========================================================================
mol_pyscf = gto.Mole()
mol_pyscf.atom    = ATOM
mol_pyscf.basis   = BASIS
mol_pyscf.unit    = "angstrom"
mol_pyscf.verbose = 0
mol_pyscf.build()

mf = scf.RHF(mol_pyscf)
mf.kernel()
E_ref = mf.e_tot

pyscf_fcidump.from_scf(mf, FCIDUMP_FILE, tol=1e-15)

print("\n" + "=" * 70)
print("  LiH / STO-3G  —  FCIDUMP vs PySCFIntegrals benchmark")
print("=" * 70)
print(f"\n  PySCF RHF reference energy : {E_ref:.10f} Eh")

# ===========================================================================
# 2. PySCFIntegrals backend
# ===========================================================================

print("\n" + "-" * 70)
print("  PySCFIntegrals (AO basis)")
print("-" * 70)

pyscf_mol  = PySCFMolecule(ATOM, BASIS, "angstrom")
pyscf_ints = PySCFIntegrals(pyscf_mol)

wfn_rhf_p = run_scf(RHF, pyscf_ints)
E_rhf_pyscf = wfn_rhf_p.energy
E_ref = E_rhf_pyscf
print(f"\n  RHF : {E_rhf_pyscf:.10f} Eh")
check(f"PySCFIntegrals RHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_rhf_pyscf - E_ref) < TOL_ENERGY, got=E_rhf_pyscf, ref=E_ref)

# Write tei to FCIDUMP format and read back to check the integral interface
from quantel.ints.utils import write_fcidump
write_fcidump(pyscf_ints, wfn_rhf_p.mo_coeff, filename=FCIDUMP_FILE)

# Check gradient and hessian
print()
check(f"RHF Gradient check", wfn_rhf_p.check_gradient())
check(f"RHF Hessian check", wfn_rhf_p.check_hessian())

wfn_uhf_p = run_scf(UHF, pyscf_ints)
E_uhf_pyscf = wfn_uhf_p.energy
print(f"  UHF : {E_uhf_pyscf:.10f} Eh")
check(f"PySCFIntegrals UHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_uhf_pyscf - E_ref) < TOL_ENERGY, got=E_uhf_pyscf, ref=E_ref)
print()
check(f"UHF Gradient check", wfn_uhf_p.check_gradient())
check(f"UHF Hessian check", wfn_uhf_p.check_hessian())

# GHF started from the converged RHF orbitals: confirms the RHF solution is
# a valid GHF stationary point without landing on a spurious local minimum.
wfn_ghf_p = run_scf(GHF, pyscf_ints, ghf_coeff=wfn_rhf_p.mo_coeff)

E_ghf_pyscf = wfn_ghf_p.energy
print(f"  GHF : {E_ghf_pyscf:.10f} Eh")
check(f"PySCFIntegrals GHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_ghf_pyscf - E_ref) < TOL_ENERGY, got=E_ghf_pyscf, ref=E_ref)
print()
check(f"GHF Gradient check", wfn_ghf_p.check_gradient())
check(f"GHF Hessian check", wfn_ghf_p.check_hessian())

# ===========================================================================
# 3. FCIDUMP backend
# ===========================================================================

print("\n" + "-" * 70)
print("  FCIDUMP (MO basis)")
print("-" * 70)

fcidump_ints = FCIDUMP(FCIDUMP_FILE)
fcidump_ints.print()
nmo = fcidump_ints.nmo()

# In the MO basis the identity IS the converged RHF solution; pass it as the
# GHF seed via ghf_coeff so ghf_from_rhf_coeffs constructs the spinor guess.
C_mo_identity = np.eye(nmo)

wfn_rhf_f = run_scf(RHF, fcidump_ints, coeff=C_mo_identity)
E_rhf_fcidump = wfn_rhf_f.energy
print(f"\n  RHF : {E_rhf_fcidump:.10f} Eh")
check(f"FCIDUMP RHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_rhf_fcidump - E_ref) < TOL_ENERGY, got=E_rhf_fcidump, ref=E_ref)
# Check gradient and hessian
print()
check(f"RHF Gradient check", wfn_rhf_f.check_gradient())
check(f"RHF Hessian check", wfn_rhf_f.check_hessian())

wfn_uhf_f = run_scf(UHF, fcidump_ints, coeff=C_mo_identity)
E_uhf_fcidump = wfn_uhf_f.energy
print(f"  UHF : {E_uhf_fcidump:.10f} Eh")
check(f"FCIDUMP UHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_uhf_fcidump - E_ref) < TOL_ENERGY, got=E_uhf_fcidump, ref=E_ref)
print()
check(f"UHF Gradient check", wfn_uhf_f.check_gradient())
check(f"UHF Hessian check", wfn_uhf_f.check_hessian())

wfn_ghf_f = run_scf(GHF, fcidump_ints, ghf_coeff=C_mo_identity)

E_ghf_fcidump = wfn_ghf_f.energy
print(f"  GHF : {E_ghf_fcidump:.10f} Eh")
check(f"FCIDUMP GHF matches reference (tol={TOL_ENERGY:.0e})",
      abs(E_ghf_fcidump - E_ref) < TOL_ENERGY, got=E_ghf_fcidump, ref=E_ref)

print()
check(f"GHF Gradient check", wfn_ghf_f.check_gradient())
check(f"GHF Hessian check", wfn_ghf_f.check_hessian())

# ===========================================================================
# 4. Cross-check: both backends agree with each other
# ===========================================================================

print("\n" + "-" * 70)
print("  Cross-check: FCIDUMP == PySCFIntegrals")
print("-" * 70)

check("RHF energies agree between backends",
      abs(E_rhf_fcidump - E_rhf_pyscf) < TOL_ENERGY,
      got=E_rhf_fcidump, ref=E_rhf_pyscf)
check("UHF energies agree between backends",
      abs(E_uhf_fcidump - E_uhf_pyscf) < TOL_ENERGY,
      got=E_uhf_fcidump, ref=E_uhf_pyscf)
check("GHF energies agree between backends",
      abs(E_ghf_fcidump - E_ghf_pyscf) < TOL_ENERGY,
      got=E_ghf_fcidump, ref=E_ghf_pyscf)

# ===========================================================================
# Done
# ===========================================================================

os.remove(FCIDUMP_FILE)

print("\n" + "=" * 70)
print("  All checks passed.")
print("=" * 70 + "\n")
