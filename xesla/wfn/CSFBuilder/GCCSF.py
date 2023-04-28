r"""
Construct CSFs via genealogical coupling
"""
import itertools
import numpy as np
from pyscf import gto, scf, fci
from typing import List
from scipy import linalg
from CSFTools.CouplingCoefficients import get_total_coupling_coefficient
from GenericCSF import GenericCSF
from Operators.Operators import create
from CSFTools.PermutationTools import get_phase_factor
from ReducedDensityMatrices.RDMapper import mapper, get_dm12


class GCCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float, core: List[int], act: List[int],
                 g_coupling: str, permutation: List[int] = None, mo_basis="site"):
        self.mol = mol
        super().__init__(stot, mol.spin / 2, len(core), len(act), mol.nelectron)
        self.g_coupling = g_coupling
        self.mo_basis = mo_basis
        self.permutation = permutation
        coeffs = self.get_coeffs(method=self.mo_basis)

        # Get information about the orbitals used
        self.core_orbs = coeffs[:, core]  # Core orbitals
        self.act_orbs = coeffs[:, act]  # Active orbitals
        if self.permutation is not None:
            self.act_orbs = self.act_orbs[:, permutation]
        self.n_orbs = self.n_core + self.n_act  # Number of spatial orbitals
        self.coeffs = np.hstack([self.core_orbs, self.act_orbs])

        self.dets_orbrep = self.form_dets_orbrep()
        self.n_dets = len(self.dets_orbrep)
        self.unique_orb_config = []
        self.det_phase_factors = np.zeros(self.n_dets)
        self.det_dict = self.form_det_dict()
        self.dets_sq = self.form_dets_sq()  # Determinants in Second Quantisation
        self.csf = self.csf_from_g_coupling(g_coupling)
        self.csf_coeffs = self.get_specific_csf_coeffs()
        self.ci = self.get_civec()

    def get_coeffs(self, method):
        r"""
        Get the coefficient matrices. Defaults to site basis (easier).
        :param method: :str: Defines the basis used.
                        "site": Site basis
                        "hf": Hartree--Fock (HF) basis
        :return: 2D np.ndarray corresponding to the coefficients (permuted based on self.permutation)
        """
        if method == 'site':
            overlap = self.mol.intor('int1e_ovlp_sph')
            return linalg.sqrtm(np.linalg.inv(overlap))
        if method == 'hf':
            # Runs RHF calculation
            mf = scf.rhf.RHF(self.mol)
            mf.scf()
            return mf.mo_coeff
        if method == 'rohf':
            mf = scf.ROHF(self.mol)
            mf.scf()
            return mf.mo_coeff
        if method == 'custom':
            print("Using custom orbitals")
            coeffs = np.load("custom_mo.npy")
            return coeffs

    def form_dets_orbrep(self):
        r"""
        Returns ALL possible permutations of alpha and beta electrons in the active orbitals given.
        :return: List[Tuple, Tuple] of the alpha and beta electrons. e.g. [(0,1), (1,2)]
                 shows alpha electrons in 0th and 1st orbitals, beta electrons in 1st and 2nd orbitals
        """
        alpha_act = list(itertools.combinations(np.arange(self.n_act), self.n_alpha - self.n_core))
        beta_act = list(itertools.combinations(np.arange(self.n_act), self.n_beta - self.n_core))
        alpha_rep = []
        beta_rep = []
        for i, tup in enumerate(alpha_act):
            alpha_rep.append(tup)
        for i, tup in enumerate(beta_act):
            beta_rep.append(tup)
        return list(itertools.product(alpha_rep, beta_rep))

    @staticmethod
    def get_det_str(singly_occ, alpha_idxs, beta_idxs):
        r"""
        Get the determinants in a M_{s} representation.
        For example, for molecular orbitals \psi_{i}, \psi_{j} and \psi_{k} with spins alpha, beta, and alpha, respectively,
        the representation will be [0, 0.5, 0, 0.5]
        This is to help with finding out CSF coefficients
        :param singly_occ: List of singly occupied orbitals
        :param alpha_idxs: List of orbitals with alpha electrons
        :param beta_idxs: List of orbitals with beta electrons
        :return: List corresponding to the determinant in M_{s} representation.
        """
        det = [0]
        for i in range(len(singly_occ)):
            if singly_occ[i] in alpha_idxs:
                det.append(det[-1] + 0.5)
            elif singly_occ[i] in beta_idxs:
                det.append(det[-1] - 0.5)
            else:
                print("You have really messed up")
        return det

    def form_det_dict(self):
        r"""
        Forms the dictionary containing all determinants involved. The key indexes the determinant.
        The value is a list. We extract all the necessary information about the determinant here.
        val[0] = DETERMINANT in orbital representation (alpha/beta orbitals)
        val[1] = [Singly occupied orbitals, doubly occupied orbitals]
        val[2] = Determinant string value representation of the singly occupied orbitals
        :return: Dictionary with determinant index as key, and List val as value
        """
        det_dict = {}
        for idx in range(self.n_dets):
            val = []
            alpha_idxs = list(self.dets_orbrep[idx][0])
            beta_idxs = list(self.dets_orbrep[idx][1])
            orb_rep = [alpha_idxs, beta_idxs]
            all_occ = list(set(alpha_idxs).union(set(beta_idxs)))
            double_occ = list(set(alpha_idxs).intersection(set(beta_idxs)))
            singly_occ = list(set(all_occ) - set(double_occ))
            orb_idxs = [singly_occ, double_occ]
            if orb_idxs not in self.unique_orb_config:
                self.unique_orb_config.append(orb_idxs)
            coupling = self.get_det_str(singly_occ, alpha_idxs, beta_idxs)
            val.append(orb_rep)
            val.append(orb_idxs)
            val.append(coupling)
            det_dict[idx] = val
            self.det_phase_factors[idx] = get_phase_factor(alpha_idxs, beta_idxs)
        return det_dict

    def create_det_sq(self, alpha_idxs, beta_idxs, idx):
        r"""
        From list of occupied alpha orbitals (alpha_idxs) and list of occupied beta orbitals (beta_idxs),
        get the determinant in second quantisation
        :param alpha_idxs:
        :param beta idxs:
        :return:
        """
        ket = [0] * (self.n_act * 2 + 1)
        ket[0] = 1
        for _, i in enumerate(beta_idxs):
            create(i + 1 + self.n_act, ket)
        for _, i in enumerate(alpha_idxs):
            create(i + 1, ket)
        ket[0] = self.det_phase_factors[idx]
        return ket

    def form_dets_sq(self):
        r"""
        Create determinant strings in second quantisation
        :return:
        """
        dets_sq = []
        for idx in range(self.n_dets):
            alpha_idxs = list(self.dets_orbrep[idx][0])
            beta_idxs = list(self.dets_orbrep[idx][1])
            dets_sq.append(self.create_det_sq(alpha_idxs, beta_idxs, idx))
        return dets_sq

    def csf_from_g_coupling(self, g_coupling):
        r"""
        Forms CSF from a genealogical coupling pattern
        :return: CSF in the format [0, 0.5, ..., ]
        """
        if g_coupling is None:
            return None
        csf = [0.0]
        g_coupling_rep = list(g_coupling)
        assert (g_coupling_rep[0] == '+')
        for i, rep in enumerate(g_coupling_rep):
            if rep == '+':
                csf.append(csf[i] + 0.5)
            else:  # rep == '-'
                csf.append(csf[i] - 0.5)
        assert np.allclose(csf[-1], self.stot, rtol=0, atol=1e-6)
        return csf

    def get_specific_csf_coeffs(self):
        r"""
        Get coefficients for a specific CSF
        :return:
        """
        csf_coeffs = np.zeros(self.n_dets)  # These are all the determinants possible (same orbital configurations)
        if self.csf is None:
            assert self.n_dets == 1
            csf_coeffs[0] = 1
            return csf_coeffs
        for d_key, d_val in self.det_dict.items():
            if len(d_val[2]) != len(self.csf):
                csf_coeffs[d_key] = 0
            else:
                csf_coeffs[d_key] = get_total_coupling_coefficient(d_val[2], self.csf)
        return csf_coeffs

    def update_coeffs(self, new_coeffs):
        r"""
        Updates MO coefficient matrices
        :return:
        """
        self.coeffs = new_coeffs

    def get_s2(self):
        r"""
        Gets the S^2 value from PySCF
        """
        fcisolver = fci.direct_spin1.FCISolver(self.mol)
        n_elec_act = self.n_elec - 2 * self.n_core
        return fcisolver.spin_square(self.ci, self.n_act, n_elec_act)[0]