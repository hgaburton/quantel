r"""
This class is responsible for the construction of CSFs
"""
import copy
import itertools
import numpy as np
from math import comb
from pyscf import gto, scf
from typing import List
from scipy import linalg
from CouplingCoefficients import get_total_coupling_coefficient
from Integrals import get_1e_int_gen, get_1e_core, get_2e
from PermutationTools import get_phase_factor

np.set_printoptions(precision=6, suppress=True)


class CSFConstructor:
    def __init__(self, mol: gto.Mole, s: float, target_csfs: List[int], permutation: List[int] = None,
                 mo_basis="custom", mo_coeffs = None):
        self.mol = mol
        self.bas = "sph"
        self.mo_basis = mo_basis
        self.n_elec = mol.nelectron  # Number of electrons
        self.spin = mol.spin  # M_s value
        self.s = s
        self.target_csfs = target_csfs
        self.n_alpha = (mol.nelectron + 2 * mol.spin) // 2  # Number of alpha electrons
        self.n_beta = (mol.nelectron - 2 * mol.spin) // 2  # Number of beta electrons
        self.e_nuc = self.mol.energy_nuc()
        self.overlap = get_1e_int_gen('int1e_ovlp_sph', self.mol)
        self.hcore = get_1e_core(self.mol, self.bas)
        self.rij_matrix = get_2e(self.mol, self.bas)
        self.permutation = permutation
        if mo_basis == 'custom':
            self.coeffs = mo_coeffs
        else:
            self.coeffs = self.get_coeffs(method=self.mo_basis)
        self.n_orbs = self.coeffs.shape[0]  # Number of spatial orbitals
        # Number of ways to arrange e in spatial orbs
        self.n_dets = comb(self.n_orbs, self.n_alpha) * comb(self.n_orbs, self.n_beta)
        # Gets the orbitals for alpha and beta electrons
        self.dets_orbrep = self.form_dets_orbrep()
        self.unique_orb_config = []
        self.det_phase_factors = np.zeros(self.n_dets)
        self.det_dict = self.form_det_dict()
        self.n_csfs = 0
        self.csfs = self.form_csfs()
        self.csf_coeffs = self.get_csf_coeffs()

    def get_coeffs(self, method):
        r"""
        Get the coefficient matrices. Defaults to site basis (easier).
        :param method: :str: Defines the basis used.
                        "site": Site basis
                        "hf": Hartree--Fock (HF) basis
                        "hf_lc": Linear combination of HF orbitals
        :return: 2D np.ndarray corresponding to the coefficients (permuted based on self.permutation)
        """
        if method == 'site':
            if self.permutation is None:
                return linalg.sqrtm(np.linalg.inv(self.overlap))
            else:
                return linalg.sqrtm(np.linalg.inv(self.overlap))[:, self.permutation]
        if method == 'hf':
            # Runs RHF calculation
            mf = scf.rhf.RHF(self.mol)
            mf.scf()
            if self.permutation is None:
                return mf.mo_coeff
            else:
                return mf.mo_coeff[:, self.permutation]

    def form_dets_orbrep(self):
        r"""
        Returns all possible permutations of alpha and beta electrons in the orbitals given.
        :return: List[Tuple, Tuple] of the alpha and beta electrons. e.g. [(0,1), (1,2)]
                 shows alpha electrons in 0th and 1st orbitals, beta electrons in 1st and 2nd orbitals
        """
        alpha_rep = list(itertools.combinations(np.arange(self.n_orbs), self.n_alpha))
        beta_rep = list(itertools.combinations(np.arange(self.n_orbs), self.n_beta))
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
            # While we are at it, let's find some phase factors
            self.det_phase_factors[idx] = get_phase_factor(alpha_idxs, beta_idxs)
        return det_dict

    def filter_csfs(self, all_csfs):
        r"""
        Filter out CSFs which do not have the same spin symmetry as what was requested.
        :param all_csfs: List of CSFs in S representation
        :return: List of CSFs in S representation, filtered for the spin symmetry defined in the CSFConstructor object
        """
        filtered_csfs = []
        for _, csf in enumerate(all_csfs):
            if np.isclose(csf[-1], self.s, rtol=0, atol=1e-10):
                filtered_csfs.append(csf)
        return filtered_csfs

    def form_csfs_orb_config(self, singly_occ):
        r"""
        Construct CSFs for each unique combination of singly occupied orbitals

        :param singly occ:
        :return:
        """
        all_csfs = [[0.0]]
        for i in range(len(singly_occ)):  # For loop to add one more entry for each orbital
            cur_csfs = []
            for csf in all_csfs:  # In each addition step, we loop over all CSFs already formed
                prev_entry = csf[-1]
                if prev_entry == 0:
                    csf.append(0.5)
                    cur_csfs.append(csf)
                else:
                    csfplus = copy.deepcopy(csf)
                    csfminus = copy.deepcopy(csf)
                    csfplus.append(prev_entry + 0.5)
                    csfminus.append(prev_entry - 0.5)
                    cur_csfs.append(csfplus)
                    cur_csfs.append(csfminus)
            all_csfs = cur_csfs
        return self.filter_csfs(all_csfs)

    def form_csfs(self):
        r"""
        For each orbital configuration, produce CSF
        :return:
        """
        all_csfs = []
        for orb_config in self.unique_orb_config:
            if not orb_config[0]:  # No singly occupied orbitals
                self.n_csfs += 1  # This is already a CSF
            else:  # Singly occupied orbitals exists
                orb_config_csfs = self.form_csfs_orb_config(orb_config[0])
                all_csfs.append([orb_config, orb_config_csfs])
                self.n_csfs += len(orb_config_csfs)
        return all_csfs

    def get_csf_coeffs(self):
        r"""
        For each CSF, beginning with the doubly occupied ones, get the coefficients
        :return:
        """
        csf_coeffs = np.zeros((self.n_dets, self.n_csfs))
        idx = 0
        for orb_config in self.unique_orb_config:
            if not orb_config[0]:  # No singly occupied orbitals
                # We look up the index of it in the dictionary
                for key, val in self.det_dict.items():
                    if val[1] == orb_config:
                        csf_coeffs[key][idx] = 1  # The determinant is the CSF
                        idx += 1
        # Iterate through orbital configurations.
        # One configuration can have multiple CSFs
        for c_i, c_vals in enumerate(self.csfs):
            # We go through the dictionary. For each orb_config, find the determinants and the coupling coefficient
            # with each CSF. Update coeff matrix accordingly.
            for d_key, d_val in self.det_dict.items():
                if d_val[1] == c_vals[0]:
                    for i, csf in enumerate(c_vals[1]):
                        csf_coeffs[d_key][i + idx] = get_total_coupling_coefficient(d_val[2], csf)
            idx += len(c_vals[1])
        return csf_coeffs

    def get_det(self, idx):
        r"""
        Returns the orbitals in AO basis of Slater determinants
        :param idx:
        :return:
        """
        state = []
        alpha_idxs = list(self.det_dict[idx][0][0])
        beta_idxs = list(self.det_dict[idx][0][1])
        alpha = self.coeffs[:, alpha_idxs]
        beta = self.coeffs[:, beta_idxs]
        # We introduce a phase to the determinant. MOs are unique only up to a phase factor. We change the
        # first to change the sign of the overall determinant.
        alpha[:, 0] *= self.det_phase_factors[idx]
        state.append(alpha)
        state.append(beta)
        return state