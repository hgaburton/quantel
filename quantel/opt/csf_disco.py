#!/usr/bin/python3

"""
CSF Discrete-Continuous Optimization (DISCO)

Each macrocycle has two phases:

  1. Continuous  — Basin-hopping search over the continuous manifold of
                   orbital rotations for the current spin coupling vector.

                   Each hop consists of:
                     (a) Apply a random orbital rotation perturbation of
                         magnitude hop_step to the current local minimum.
                     (b) Minimize with L-BFGS to reach the next local minimum.
                     (c) Accept or reject via Metropolis:
                           P = exp(−ΔE / hop_temperature)
                         Always accept if ΔE < 0.
                   The global lowest minimum found across all hops is
                   retained at the end of phase 1.

  2. Discrete    — Combined local search in the space of spin coupling
                   vectors and active-orbital orderings. For each
                   neighbour coupling (reachable by one of the moves below),
                   csf_reorder_orbitals() is used to find the best orbital
                   ordering for that coupling before the energy is evaluated.
                   The (coupling, ordering) pair with the lowest energy is
                   the candidate move.

                   Move types tried from the current coupling string:
                       (a) Swap one '+' and one '-' at different positions.
                       (b) Insert or delete a balanced '+'/'-' pair.

                   The best downhill candidate is always accepted. If no
                   downhill move exists, the best uphill candidate is
                   accepted with Metropolis probability
                       P = exp(−ΔE / T)
                   where T is a user-supplied fictitious temperature.

Valid spin coupling vectors satisfy the genealogical (ballot) condition:
every prefix has a non-negative cumulative spin sum
('+' contributes +1, '-' contributes −1).
"""

import io
import sys
import contextlib
import numpy as np
from quantel.opt.lbfgs import LBFGS
from quantel.utils.csf_utils import csf_reorder_orbitals, get_vector_coupling


# ---------------------------------------------------------------------------
# Spin coupling neighbourhood generators
# ---------------------------------------------------------------------------

def _is_valid_coupling(sc):
    """Return True iff *sc* satisfies the genealogical ballot condition."""
    cumsum = 0
    for c in sc:
        if c == '+':
            cumsum += 1
        else:
            cumsum -= 1
            if cumsum < 0:
                return False
    return True


def _prefix_sums(sc):
    """Return list of length len(sc)+1 where entry k is the cumulative +1/-1 sum of sc[:k]."""
    prefix = [0] * (len(sc) + 1)
    for k, c in enumerate(sc):
        prefix[k + 1] = prefix[k] + (1 if c == '+' else -1)
    return prefix


def valid_spin_couplings(nopen, nalfa):
    """Return all valid spin coupling strings with *nopen* positions and *nalfa* '+' characters.

    Parameters
    ----------
    nopen : int
    nalfa : int

    Returns
    -------
    list[str]
    """
    if nopen == 0:
        return [""]
    results = []
    _enumerate_couplings(nopen, nalfa, nopen - nalfa, 0, "", results)
    return sorted(results)


def _enumerate_couplings(remaining, nalfa_left, nbeta_left, cumsum, current, out):
    if remaining == 0:
        out.append(current)
        return
    if nalfa_left > 0:
        _enumerate_couplings(remaining - 1, nalfa_left - 1, nbeta_left,
                             cumsum + 1, current + "+", out)
    if nbeta_left > 0 and cumsum > 0:
        _enumerate_couplings(remaining - 1, nalfa_left, nbeta_left - 1,
                             cumsum - 1, current + "-", out)


def _swap_moves(sc, prefix):
    """All unique valid couplings reachable by swapping one '+' and one '-'.

    When the '+' moves rightward (i < j) prefix sums in (i, j] drop by 2;
    valid iff the minimum in that range is >= 2.  When the '+' moves leftward
    (i > j) prefix sums in (j, i] rise by 2 — always valid.
    """
    plus_pos  = [i for i, c in enumerate(sc) if c == '+']
    minus_pos = [i for i, c in enumerate(sc) if c == '-']
    if not plus_pos or not minus_pos:
        return set()
    seen = set()
    for i in plus_pos:
        for j in minus_pos:
            if i < j:
                # Prefix sums in (i, j] drop by 2 — valid iff min >= 2.
                if min(prefix[i + 1 : j + 1]) < 2:
                    continue
                candidate = sc[:i] + '-' + sc[i + 1:j] + '+' + sc[j + 1:]
            else:
                # Prefix sums in (j, i] rise by 2 — always valid.
                candidate = sc[:j] + '+' + sc[j + 1:i] + '-' + sc[i + 1:]
            seen.add(candidate)
    return seen


def _add_pair_moves(sc, prefix):
    """All unique valid couplings with one extra '+' and one extra '-' inserted.

    When '+' is inserted before '-' (j > i in the extended string) prefix sums
    in (i, j] rise by 1 — always valid.  When '-' comes first (j <= i) prefix
    sums in [j, i] drop by 1; valid iff the minimum in that range is >= 1.

    Deduplication in the j > i branch:
      - Skip insertion point i if sc[i-1] == '+': inserting '+' within a run of
        consecutive '+'s gives the same result as inserting before the run.
      - Skip insertion point j if j > i+1 and sc[j-2] == '-': inserting '-'
        within a run of consecutive '-'s gives the same result as inserting
        before the run.
    """
    n = len(sc)
    seen = set()

    # j > i: '+' before '-' — always ballot-valid.
    for i in range(n + 1):
        if i > 0 and sc[i - 1] == '+':
            continue   # duplicate of i-1 for every j > i
        for j in range(i + 1, n + 2):
            if j > i + 1 and sc[j - 2] == '-':
                continue   # duplicate of j-1 for the same i
            seen.add(sc[:i] + '+' + sc[i:j - 1] + '-' + sc[j - 1:])

    # j <= i: '-' before '+' — valid iff min(prefix[j:i+1]) >= 1.
    for i in range(n + 1):
        for j in range(i + 1):
            if min(prefix[j : i + 1]) >= 1:
                seen.add(sc[:j] + '-' + sc[j:i] + '+' + sc[i:])

    return seen


def _delete_pair_moves(sc, prefix):
    """All unique valid couplings with one '+' and one '-' removed.

    When '+' is removed before '-' (i < j) prefix sums in (i, j] drop by 1;
    valid iff the minimum in that range is >= 1.  When '-' is removed first
    (j < i) prefix sums in (j, i] rise by 1 — always valid.
    """
    if len(sc) < 2:
        return set()
    plus_pos  = [i for i, c in enumerate(sc) if c == '+']
    minus_pos = [i for i, c in enumerate(sc) if c == '-']
    seen = set()
    for i in plus_pos:
        for j in minus_pos:
            if i < j:
                # Prefix sums in (i, j] drop by 1 — valid iff min >= 1.
                if min(prefix[i + 1 : j + 1]) < 1:
                    continue
                candidate = sc[:i] + sc[i + 1:j] + sc[j + 1:]
            else:
                # Prefix sums in (j, i] rise by 1 — always valid.
                candidate = sc[:j] + sc[j + 1:i] + sc[i + 1:]
            seen.add(candidate)
    return seen


def coupling_neighbours(sc):
    """Union of all local coupling moves from *sc* (excluding *sc* itself)."""
    prefix = _prefix_sums(sc)
    neighbours = _swap_moves(sc, prefix) | _add_pair_moves(sc, prefix) | _delete_pair_moves(sc, prefix)
    neighbours.discard(sc)
    return sorted(neighbours)


# ---------------------------------------------------------------------------
# Main DISCO optimiser
# ---------------------------------------------------------------------------

class CSFDisco:
    """Discrete-Continuous Optimization (DISCO) for CSF wave functions.

    Parameters
    ----------
    maxit : int
        Maximum number of outer macrocycles (default 50).
    ethresh : float
        Outer convergence threshold on the energy change (default 1e-8 Eh).
    gthresh : float
        Gradient convergence threshold for L-BFGS (default 1e-6).
    temperature : float
        Fictitious temperature for Metropolis acceptance of uphill discrete
        coupling moves (default 0.1 Eh).  Set to 0 for strictly downhill.
    n_hop : int
        Number of basin-hopping perturbations per macrocycle in phase 1
        (default 0, i.e. plain L-BFGS with no hopping).
    hop_step : float
        RMS magnitude of the random orbital rotation applied as a
        basin-hopping perturbation (default 0.3 radians).
    hop_temperature : float
        Metropolis temperature for accepting basin-hopping moves in the
        continuous space (default 0.01 Eh).  Set to 0 to accept only
        downhill hops.
    plev : int
        Print level: 0 = silent, 1 = normal, 2 = verbose (default 1).
    lbfgs_kwargs : dict
        Extra keyword arguments forwarded to :class:`LBFGS`.
    """

    def __init__(self, **kwargs):
        self.maxit           = kwargs.get("maxit", 50)
        self.ethresh         = kwargs.get("ethresh", 1e-8)
        self.gthresh         = kwargs.get("gthresh", 1e-6)
        self.temperature     = kwargs.get("temperature", 0.5)
        self.n_hop           = kwargs.get("n_hop", 10)
        self.hop_step        = kwargs.get("hop_step", 0.3)
        self.hop_temperature = kwargs.get("hop_temperature", 0.05)
        self.select_temperature = kwargs.get("select_temperature", 0.1)
        self.adapt_rate      = kwargs.get("adapt_rate", 0.1)
        self.target_ratio    = kwargs.get("target_ratio", 0.5)
        self.taboo_tenure    = kwargs.get("taboo_tenure", 5)
        self.plev            = kwargs.get("plev", 1)
        self.lbfgs           = LBFGS(**kwargs.get("lbfgs_kwargs", {}))

    # ------------------------------------------------------------------
    # Adaptive temperature
    # ------------------------------------------------------------------

    def _adapt_temperature(self, T, ema, accepted):
        """Update temperature via exponential moving average of acceptance.

        Parameters
        ----------
        T : float
            Current temperature.
        ema : float
            Current EMA of the acceptance signal (in [0, 1]).
        accepted : bool
            Whether the latest Metropolis step was accepted.

        Returns
        -------
        T_new : float
        ema_new : float
        """
        rate    = self.adapt_rate
        target  = self.target_ratio
        ema_new = (1.0 - rate) * ema + rate * float(accepted)
        T_new   = T / np.exp(rate * (ema_new - target))
        return max(T_new, 1e-10), ema_new

    # ------------------------------------------------------------------
    # Phase 1 — basin-hopping over the continuous orbital manifold
    # ------------------------------------------------------------------

    def _basin_hop_phase1(self, csf, lbfgs_maxit, lbfgs_plev):
        """Basin-hopping search over orbital rotations for the current coupling.

        Starting from the current state of *csf*:
          1. Minimize with L-BFGS to reach a local minimum (the reference).
          2. For each of the *n_hop* hops:
               a. Perturb from the current accepted minimum with a random
                  orbital rotation of RMS magnitude *hop_step*.
               b. Minimize with L-BFGS.
               c. Accept via Metropolis(hop_temperature); always accept if
                  ΔE < 0.
          3. Restore *csf* to the lowest minimum found across all hops.
          4. After each uphill Metropolis decision, adapt *hop_temperature*
             so that the acceptance ratio converges to *target_ratio*.

        Parameters
        ----------
        hop_ema : float
            Running EMA of uphill-hop acceptance from previous macrocycles.

        Returns
        -------
        e_best : float
        n_accepted : int
        minima : list
        hop_ema : float
            Updated EMA after this macrocycle's hops.
        """
        plev = self.plev
        sc   = csf.spin_coupling

        # ── Initial minimization ─────────────────────────────────────────
        self.lbfgs.run(csf, thresh=self.gthresh, maxit=lbfgs_maxit, plev=0)
        if self._is_new_minimum(csf):
            # Record every converged minimum regardless of acceptance.
            if plev > 1: 
                print(f"  New minimum with energy = {csf.energy: .10f} Eh and spin coupling = {csf.spin_coupling!r}")
            self.all_minima.append((csf.energy, csf.copy()))
        e_best     = csf.energy
        mo_best    = csf.mo_coeff.copy()
        e_current  = e_best
        mo_current = mo_best.copy()

        if plev > 0:
            print(f"  Hop   0 (initial):  E = {e_best: .10f}  T_hop={self.hop_temperature:.4e}")
            sys.stdout.flush()

        if self.n_hop == 0:
            return e_best, 0

        n_accepted = 0
        for ihop in range(1, self.n_hop + 1):
            # ── Perturbation ─────────────────────────────────────────────
            csf.initialise(mo_current, sc)
            perturb = np.random.randn(csf.nrot)
            perturb *= self.hop_step / (np.linalg.norm(perturb) / np.sqrt(csf.nrot))
            csf.take_step(perturb)

            # ── Minimization ─────────────────────────────────────────────
            self.lbfgs.run(csf, thresh=self.gthresh, maxit=lbfgs_maxit, plev=0)
            e_hop = csf.energy
            delta = e_hop - e_current
            # Check to see if we have a distinct minimum
            if self._is_new_minimum(csf):
                # Record every converged minimum regardless of acceptance.
                print(f"  New minimum recorded with energy = {e_hop: .10f} Eh and spin coupling = {sc!r}") if plev > 1 else None
                self.all_minima.append((csf.energy, csf.copy()))

            # ── Metropolis acceptance ─────────────────────────────────────
            if delta < 0.0:
                accepted = True
                comment  = f"downhill  ΔE={delta: .3e}"
            elif self.hop_temperature > 0.0:
                prob     = np.exp(-delta / self.hop_temperature)
                accepted = np.random.rand() < prob
                comment  = f"{'accepted' if accepted else 'rejected'} Metropolis  ΔE=+{delta:.3e}  P={prob:.3f}"
            else:
                accepted = False
                comment  = f"rejected  ΔE=+{delta:.3e}"

            if accepted:
                e_current  = e_hop
                mo_current = csf.mo_coeff.copy()
                n_accepted += 1
                if e_hop < e_best:
                    e_best  = e_hop
                    mo_best = csf.mo_coeff.copy()

            # ── Adapt hop_temperature (uphill steps only) ─────────────────
            if delta >= 0.0:
                self.hop_temperature, self.hop_ema = self._adapt_temperature(self.hop_temperature, self.hop_ema, accepted)

            if plev > 0:
                tag = "*" if e_hop < e_best + 1e-12 else " "
                print(f"  Hop {ihop:3d}{tag}:  E = {e_hop: .10f}  {comment}  "
                      f"T_hop={self.hop_temperature:.4e}")
                sys.stdout.flush()

        # Restore to the global best minimum found across all hops.
        csf.initialise(mo_best, sc)
        return e_best, n_accepted

    # ------------------------------------------------------------------
    # Minimum deduplication by overlap
    # ------------------------------------------------------------------

    def _is_new_minimum(self, csf, ovlp_thresh=1e-6):
        """Return True if (mo, sc) is distinct from every minimum in *existing*.

        Two minima are considered the same if |<i|j>| > ovlp_thresh.

        Parameters
        ----------
        csf : CSF
            Will be temporarily reinitialised; restored on return.
        mo : ndarray
        sc : str
        existing : list of (energy, mo, sc)
        ovlp_thresh : float
        """
        if not self.all_minima:
            return True
        for ej,csfj in self.all_minima:
            # Compare energies
            if(abs(csf.energy - csfj.energy) < 1e-10):
                return False
            # Compare overlap
            if(abs(1-csf.overlap(csfj)) < ovlp_thresh):
                return False
        return True

    # ------------------------------------------------------------------
    # Phase 2 — combined discrete coupling + orbital reordering search
    # ------------------------------------------------------------------

    def _best_ordering_for_coupling(self, csf, mo_coeff, sc):
        """Return mo_coeff with active orbitals reordered for coupling *sc*.

        Uses csf_reorder_orbitals to localise and permute the active orbitals
        to minimise the exchange energy for the given coupling.  For couplings
        with fewer than 2 active orbitals no reordering is possible and
        mo_coeff is returned unchanged.

        Parameters
        ----------
        csf : CSF
        mo_coeff : ndarray, shape (nbsf, nmo)
        sc : str
            Target spin coupling string.

        Returns
        -------
        ndarray, shape (nbsf, nmo)
            Copy of mo_coeff with reordered active columns.
        """
        nopen_sc = len(sc)
        ncore    = csf.ncore
        nocc_sc  = ncore + nopen_sc

        mo_try = mo_coeff.copy()

        if nopen_sc < 2:
            return mo_try

        # Active orbitals for this coupling (may include virtuals for add-pair moves)
        cinit = mo_try[:, ncore:nocc_sc]

        # Exchange coupling matrix for sc: shape (nopen_sc, nopen_sc)
        _, bij = get_vector_coupling(csf.nmo, ncore, nocc_sc, sc)
        exchange_matrix = bij[ncore:nocc_sc, ncore:nocc_sc]

        # Suppress the verbose prints from csf_reorder_orbitals unless plev >= 2
        if self.plev < 2:
            with contextlib.redirect_stdout(io.StringIO()):
                cinit_opt = csf_reorder_orbitals(csf.integrals, exchange_matrix, cinit)
        else:
            cinit_opt = csf_reorder_orbitals(csf.integrals, exchange_matrix, cinit)

        mo_try[:, ncore:nocc_sc] = cinit_opt
        return mo_try

    def _eval_coupling_neighbours(self, csf, mo_coeff, sc_current, e_current):
        """Evaluate every valid coupling neighbour with optimal orbital ordering.

        For each neighbour coupling, csf_reorder_orbitals is called to find
        the best active-orbital ordering before the energy is evaluated.
        Boltzmann weights (exp(-ΔE/T) / Z) relative to *e_current* are
        computed so they can be reported and used for sampling.
        After this call *csf* is left in an arbitrary state; the caller is
        responsible for reinitialising *csf* to the chosen coupling.

        Returns
        -------
        list[tuple[float, str, ndarray, float]]
            Sorted (ascending energy) list of (energy, coupling, mo_coeff,
            boltzmann_weight) tuples.
        """
        neighbours = coupling_neighbours(sc_current)
        results = []
        for sc in neighbours:
            mo_try = self._best_ordering_for_coupling(csf, mo_coeff, sc)
            csf.initialise(mo_try, sc)
            self.lbfgs.run(csf, thresh=self.gthresh, plev=0)
            results.append((csf.energy, sc, mo_try))
        results.sort(key=lambda x: x[0])
        return results


    def _coupling_step(self, candidate, e_current):
        """Select a discrete move from pre-computed neighbour evaluations.

        Parameters
        ----------
        results : list[tuple[float, str, ndarray]]
            Pre-computed sorted triples from :meth:`_eval_coupling_neighbours`.
        e_current : float
            Energy of the current (reference) state.

        Returns
        -------
        accepted : bool
        new_coupling : str or None
        new_mo : ndarray or None
        comment : str
        """
        best_e, best_sc, best_mo = candidate
        delta_e = best_e - e_current

        if delta_e < 0.0:
            # Downhill — always take the lowest energy neighbour.
            return True, best_sc, best_mo, best_e, f"downhill  ΔE={delta_e: .3e}"
        else: 
            # Step 2 — standard Metropolis accept/reject based on ΔE / T.
            metro_prob = np.exp(-delta_e / self.temperature)
            if np.random.rand() < metro_prob:
                return True, best_sc, best_mo, best_e, (
                    f"Metropolis  ΔE=+{delta_e:.3e}  "
                    f"w={1.0:.4f}  P_metro={metro_prob:.4f}")
            else:
                return False, None, None, best_e, (
                    f"rejected  ΔE=+{delta_e:.3e}  "
                    f"w={1.0:.4f}  P_metro={metro_prob:.4f}")

        return False, None, None, best_e, f"rejected  ΔE=+{delta_e:.3e}"

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _print_header(self, csf):
        print()
        print("  ================================================================")
        print("  CSF Discrete-Continuous Optimization (DISCO)")
        print("  ================================================================")
        print(f"  Active orbitals  : {csf.nopen}")
        print(f"  Alpha / Beta     : {csf.cas_nalfa} / {csf.cas_nbeta}")
        print(f"  Sz               : {csf.sz: .1f}")
        print(f"  Basin hops       : {self.n_hop}")
        print(f"  Hop step (RMS)   : {self.hop_step:.2e} rad")
        print(f"  Hop temperature  : {self.hop_temperature:.2e} Eh")
        print(f"  Discrete temp.   : {self.temperature:.2e} Eh")
        print(f"  Gradient thresh  : {self.gthresh:.2e}")
        print(f"  Energy thresh    : {self.ethresh:.2e}")
        print(f"  Max macrocycles  : {self.maxit}")
        print(f"  Taboo tenure     : {self.taboo_tenure}")
        print("  ================================================================")
        sys.stdout.flush()

    def _print_minima_summary(self, csf, minima):
        """Print the overlap-deduplicated energy table and pairwise overlap matrix."""
        if not minima:
            return
        # Minima are already deduplicated by overlap; just sort by energy.
        unique   = sorted(minima, key=lambda x: x[0])
        n        = len(unique)
        e_global = unique[0][0]

        # ── Energy table ─────────────────────────────────────────────────
        print()
        print("  ================================================================")
        print(f"  All distinct L-BFGS minima found  ({n} unique)")
        print("  ================================================================")
        print(f"  {'#':>4s}  {'Coupling':>14s}  {'Energy / Eh':>16s}  {'ΔE from best / Eh':>18s}")
        print(f"  {'─'*4}  {'─'*14}  {'─'*16}  {'─'*18}")
        for idx, (e, csf) in enumerate(unique):
            tag = "  <-- global min" if idx == 0 else ""
            print(f"  {idx+1:4d}  {csf.spin_coupling!r:>14s}  {e: 16.10f}  {e - e_global: 18.6e}{tag}")

        sys.stdout.flush()
        return


    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------

    def run(self, csf, lbfgs_maxit=200):
        """Run the DISCO macrocycle.

        Parameters
        ----------
        csf : CSF
            An initialized CSF wave-function object.
        lbfgs_maxit : int
            Maximum L-BFGS iterations per macrocycle (default 200).

        Returns
        -------
        converged : bool
        """
        plev  = self.plev

        if plev > 0:
            self._print_header(csf)

        e_prev     = np.inf
        sc_prev    = None
        converged  = False
        lbfgs_plev = max(0, plev - 1)
        self.all_minima = []   # collects (energy, mo_coeff, spin_coupling) for every L-BFGS minimum
        self.hop_ema    = self.target_ratio   # EMA for continuous hop acceptance
        disc_ema   = self.target_ratio   # EMA for discrete coupling acceptance
        # taboo[sc] = macrocycle at which sc was last visited
        taboo      = {}
        e_best_all = np.inf
        sc_best_all = None

        for outer_it in range(1, self.maxit + 1):

            if plev > 0:
                print(f"\n  -- Macrocycle {outer_it}  "
                      f"[coupling: {csf.spin_coupling!r}] --")
                sys.stdout.flush()

            # ── Phase 1: basin-hopping over orbital rotations ─────────────
            if plev > 0:
                hop_tag = f"  ({self.n_hop} hop(s))" if self.n_hop > 0 else ""
                print(f"\n  Phase 1 — L-BFGS orbital optimization{hop_tag}")
            e_phase1, n_accepted = self._basin_hop_phase1(csf, lbfgs_maxit, lbfgs_plev)

            if plev > 0 and self.n_hop > 0:
                print(f"  Phase 1 best:  E = {e_phase1: .10f} Eh  "
                      f"({n_accepted}/{self.n_hop} hop(s) accepted)  "
                      f"T_hop={self.hop_temperature:.4e}")
                sys.stdout.flush()

            # ── Phase 2: combined discrete coupling + reordering search ───
            if plev > 0:
                print(f"\n  Phase 2 — Discrete coupling + orbital reordering search")
            mo_after_lbfgs = csf.mo_coeff.copy()
            sc_before      = csf.spin_coupling
            e_before       = csf.energy

            # Remove taboo-expired entries before filtering neighbours.
            taboo = {sc: it for sc, it in taboo.items()
                     if outer_it - it < self.taboo_tenure}

            neighbours = coupling_neighbours(sc_before)
            # Filter out any coupling currently on the taboo list.
            allowed = [sc for sc in neighbours if sc not in taboo]

            if allowed:
                if plev > 0:
                    n_tabooed = len(neighbours) - len(allowed)
                    taboo_note = f"  ({n_tabooed} tabooed)" if n_tabooed else ""
                    print(f"  Evaluating {len(allowed)} neighbour coupling(s) with orbital reordering...{taboo_note}")
                    sys.stdout.flush()

                # Single evaluation pass — reordering is applied inside.
                results = self._eval_coupling_neighbours(csf, mo_after_lbfgs, sc_before, e_before)
                # Keep only results for allowed couplings (weights will be renormalised inside).
                results = [(e, sc, mo) for e, sc, mo in results if sc in allowed]
                
                ## Choose candidate to consider for step
                if(results[0][0] < e_before):
                    weights = np.zeros(len(results))
                    weights[0] = 1.0
                else:
                    deltas = np.array([e - e_before for e, _, _ in results])
                    weights = np.exp(-deltas / self.select_temperature)
                    weights /= weights.sum()
                idx = np.random.choice(len(results), p=weights)
                candidate = results[idx]
                
                ## Report coupling
                if plev > 0:
                    print(f"    {'Coupling':>14s}  {'Energy / Eh':>16s}  {'Delta E / Eh':>14s}  {'Selection weight':>11s}")
                    print(f"    {'─'*14}  {'─'*16}  {'─'*14}  {'─'*11}")
                    for i in range(len(results)):
                        e_sc, sc, _ = results[i]
                        w = weights[i]
                        tag = " <-- best" if sc == results[0][1] else ""
                        print(f"    {sc:>14s}  {e_sc: 16.10f}  {e_sc - e_before: 14.6e}  {w: 11.4f}{tag}")
                    sys.stdout.flush()
                
                ## Decide whether to take the step
                accepted, sc_new, mo_new, best_e, comment = self._coupling_step(candidate, e_before)

                # Adapt discrete temperature for uphill Metropolis decisions only.
                if best_e >= e_before:
                    self.temperature, disc_ema = self._adapt_temperature(
                        self.temperature, disc_ema, accepted)

                # Reinitialise to the decided (coupling, ordering) pair.
                if accepted:
                    csf.initialise(mo_new, sc_new)
                    taboo[sc_before] = outer_it   # mark the coupling we just left
                    if plev > 0:
                        print(f"  accepted: {sc_before!r} → {candidate[1]!r}  "
                              f"E = {csf.energy: .10f}  {comment}  "
                              f"T_disc={self.temperature:.4e}")
                    # Reconverge with L-BFGS to clean up any residual orbital rotation from the reordering step.
                    self.lbfgs.run(csf, thresh=self.gthresh, maxit=lbfgs_maxit, plev=0)
                    if self._is_new_minimum(csf):
                        # Record every converged minimum regardless of acceptance.
                        if plev > 1:
                            print(f"  New minimum with energy = {csf.energy: .10f} Eh and spin coupling = {csf.spin_coupling!r}")
                        self.all_minima.append((csf.energy, csf.copy()))
                else:
                    csf.initialise(mo_after_lbfgs, sc_before)
                    if plev > 0:
                        print(f"  rejected: tested {candidate[1]!r}, kept {sc_before!r}  "
                              f"E = {csf.energy: .10f}  {comment}  "
                              f"T_disc={self.temperature:.4e}")
            elif neighbours:
                if plev > 0:
                    print(f"  All {len(neighbours)} neighbour coupling(s) are tabooed  "
                          f"[tenure={self.taboo_tenure}]")
            else:
                if plev > 0:
                    print(f"  No coupling neighbours for {sc_before!r}")

            # ── Convergence check and best-minimum tracking ───────────────
            # Find best minimum so far
            e_best, csf_best = sorted(self.all_minima, key=lambda x: x[0])[0] if self.all_minima else (csf.energy, csf)

            if plev > 0:
                taboo_str = ", ".join(
                    f"{sc!r}({self.taboo_tenure - (outer_it - it)})" for sc, it in taboo.items())
                print(f"\n  End of macrocycle {outer_it}: E = {csf.energy: .10f}  {csf.spin_coupling!r}  ")
                print(f"  Global best so far:  E = {e_best: .10f}  coupling = {csf_best.spin_coupling!r}")
                if taboo:
                    print(f"  Taboo list:  {taboo_str}  (cycles remaining)")
                sys.stdout.flush()


        if plev > 0:
            status = "converged" if converged else "did NOT converge"
            print()
            print("  ================================================================")
            print(f"  DISCO {status} in {outer_it} macrocycle(s)")
            print(f"  Final spin coupling : {csf.spin_coupling!r}")
            print(f"  Final energy        : {csf.energy: .10f} Eh")
            print("  ================================================================")
            sys.stdout.flush()
            self._print_minima_summary(csf, self.all_minima)

        return converged
