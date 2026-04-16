#!/usr/bin/env python3
"""
Example: CSF Discrete-Continuous Optimization (DISCO)
======================================================

Demonstrates the CSF-DISCO algorithm on the allyl radical (C3H5, doublet).

The allyl π system provides a natural 3-electron / 3-orbital active space
(cas_nalfa=2, cas_nbeta=1) with two valid spin coupling vectors:

    "++-"   sums = [1, 2, 1]   (high-spin intermediate)
    "+-+"   sums = [1, 0, 1]   (low-spin intermediate)

DISCO alternates between:
  1. Continuous phase  — L-BFGS minimization of orbital rotation parameters
     for the current (fixed) spin coupling.
  2. Discrete phase    — evaluation of the CSF energy for every valid spin
     coupling at the current optimal orbitals; best coupling is accepted.

Usage
-----
    python test_csf_disco.py

Run from the examples/ directory so that the mol/ path resolves correctly.
"""

import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF
from quantel.opt.csf_disco import CSFDisco, valid_spin_couplings
from quantel.drivers.noci import noci, selected_noci

np.set_printoptions(linewidth=120, precision=8, suppress=True)


# ---------------------------------------------------------------------------
# 1.  Enumerate valid spin couplings for a few representative cases
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Valid spin coupling vectors")
print("=" * 60)

cases = [
    # (nopen, nalfa, description)
    (2, 1, "open-shell singlet (nopen=2, Sz=0)"),
    (3, 2, "doublet radical   (nopen=3, Sz=1/2)"),
    (4, 2, "open-shell singlet (nopen=4, Sz=0)"),
    (5, 3, "doublet radical   (nopen=5, Sz=1/2)"),
]

for nopen, nalfa, desc in cases:
    couplings = valid_spin_couplings(nopen, nalfa)
    print(f"\n  {desc}")
    print(f"  nopen={nopen}, nalfa={nalfa}, nbeta={nopen-nalfa}")
    print(f"  {len(couplings)} valid coupling(s): {couplings}")

print()


# ---------------------------------------------------------------------------
# 2.  Set up the allyl radical
# ---------------------------------------------------------------------------
# Allyl radical: C3H5, charge=0, multiplicity=2 (spin = 2S = 1).
# Active space: 3 pi electrons in 3 pi orbitals.
#   cas_nalfa = 2,  cas_nbeta = 1  →  Sz = 1/2
#   nopen = 3  →  valid couplings: ["++-", "+-+"]

print("=" * 60)
print("  CSF-DISCO on the allyl radical")
print("  Basis: STO-3G | Active space: (3e, 3o)")
print("=" * 60)

mol  = PySCFMolecule("mol/allyl.xyz", "sto-3g", "angstrom", spin=1)
ints = PySCFIntegrals(mol)

# ---------------------------------------------------------------------------
# 3.  Run DISCO starting from each valid spin coupling
# ---------------------------------------------------------------------------

initial_couplings = valid_spin_couplings(nopen=9, nalfa=5)
print(initial_couplings)

results = []   # collect (energy, final_coupling, initial_coupling)

for init_sc in ['+++']:#initial_couplings:
    print(f"\n{'─'*60}")
    print(f"  Initial spin coupling: {init_sc!r}")
    print(f"{'─'*60}")

    wfn = CSF(ints, init_sc)
    print(init_sc)
    wfn.get_orbital_guess(method="gwh")

    disco = CSFDisco(
        maxit=10,
        ethresh=1e-8,
        gthresh=1e-6,
        plev=5,
        n_hop=20,
        taboo_tenure=10,
        lbfgs_kwargs=dict(
            maxstep=0.5,
            with_transport=True,
            with_canonical=True,
        ),
    )

    print(wfn.spin_coupling)

    converged = disco.run(wfn, lbfgs_maxit=200)

    results.append({
        "init_coupling": init_sc,
        "final_coupling": wfn.spin_coupling,
        "energy": wfn.energy,
        "converged": converged,
    })

    # ---------------------------------------------------------------------------
    # NOCI using all distinct minima found by DISCO
    # ---------------------------------------------------------------------------

    minima = sorted(disco.all_minima, key=lambda x: x[0])

    for nstate in [len(minima)]:
        statelist = minima[:nstate+1]
        print(f"\n{'═'*60}")
        print(f"  NOCI on {nstate} distinct minima from DISCO")
        print(f"{'═'*60}")
        print(f"  {'#':>4s}  {'Coupling':>10s}  {'Energy / Eh':>16s}  {'ΔE / mEh':>10s}")
        print(f"  {'─'*4}  {'─'*10}  {'─'*16}  {'─'*10}")
        e_best = minima[0][0]
        for idx, (e, csf_min) in enumerate(statelist):
            print(f"  {idx+1:4d}  {csf_min.spin_coupling!r:>10s}  {e: 16.10f}  {(e-e_best)*1000: 10.4f}")

        wfn_list = [csf_min for _, csf_min in statelist]
        Hwx, Swx, eigval, v = selected_noci(wfn_list, plev=1)

# ---------------------------------------------------------------------------
# 4.  Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  DISCO summary")
print("=" * 60)
print(f"  {'Init coupling':>14s}  {'Final coupling':>14s}  {'Energy / Eh':>16s}  {'Conv':>5s}")
print(f"  {'─'*14}  {'─'*14}  {'─'*16}  {'─'*5}")
for r in results:
    print(f"  {r['init_coupling']:>14s}  {r['final_coupling']:>14s}  "
          f"{r['energy']: 16.10f}  {'yes' if r['converged'] else 'no':>5s}")

energies = [r["energy"] for r in results]
best_idx = int(np.argmin(energies))
print(f"\n  Global lowest energy : {energies[best_idx]: .10f} Eh")
print(f"  Best spin coupling   : {results[best_idx]['final_coupling']!r}")
print()
