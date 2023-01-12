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
from csfs.Auxiliary.SpatialBasis import spatial_one_and_two_e_int
from csfs.ConfigurationStateFunctions.CouplingCoefficients import get_total_coupling_coefficient
from csfs.Operators.Operators import create
from csfs.ConfigurationStateFunctions.PermutationTools import get_phase_factor
from csfs.ReducedDensityMatrices.ReducedDensityMatrices import get_mc_one_rdm, get_ri_mc_two_rdm,\
    get_spatial_one_rdm, get_spatial_two_rdm


np.set_printoptions(precision=6, suppress=True)


class ConfigurationStateFunction:
    def __init__(self, mol: gto.Mole, s: float, permutation: List[int] = None, mo_basis="site"):
        self.mol = mol
        self.mo_basis = mo_basis
        self.n_elec = mol.nelectron  # Number of electrons
        self.spin = mol.spin  # M_s value
        self.s = s
        self.n_alpha = (mol.nelectron + 2 * mol.spin) // 2  # Number of alpha electrons
        self.n_beta = (mol.nelectron - 2 * mol.spin) // 2  # Number of beta electrons
        self.permutation = permutation
        self.coeffs = self.get_coeffs(method=self.mo_basis)
        # Get MO integrals here
        self.hcore, self.eri = spatial_one_and_two_e_int(self.mol, self.coeffs) 
        self.n_orbs = self.coeffs.shape[0]  # Number of spatial orbitals
        # Number of ways to arrange e in spatial orbs
        self.n_dets = comb(self.n_orbs, self.n_alpha) * comb(self.n_orbs, self.n_beta)
        # Gets the orbitals for alpha and beta electrons
        self.dets_orbrep = self.form_dets_orbrep()
        self.unique_orb_config = []
        self.det_phase_factors = np.zeros(self.n_dets)
        self.det_dict = self.form_det_dict()
        self.dets_sq = self.form_dets_sq()  # Determinants in Second Quantisation
        self.n_csfs = 0
        self.csfs = self.form_csfs()
        self.csf_coeffs = self.get_csf_coeffs()

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
            if self.permutation is None:
                return linalg.sqrtm(np.linalg.inv(overlap))
            else:
                return linalg.sqrtm(np.linalg.inv(overlap))[:, self.permutation]
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
        ket = [0] * (self.n_orbs * 2 + 1)
        ket[0] = 1
        for _, i in enumerate(beta_idxs):
            create(i+1+self.n_orbs, ket)
        for _, i in enumerate(alpha_idxs):
            create(i+1, ket)
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

    def update_coeffs(self, new_coeffs):
        r"""
        Updates MO coefficient matrices
        :return:
        """
        self.coeffs = new_coeffs

    @staticmethod
    def get_relevant_dets(dets, coeffs, thresh=1e-10):
        r"""
        Filters a list of determinants (in Second Quantised representation)
        based on the coefficients. If coefficients are smaller than the given threshold, ignore
        """
        filtered_dets = []
        filtered_coeffs = []
        for i, coeff in enumerate(coeffs):
            if np.isclose(coeff, 0, rtol=0, atol=thresh):
                pass
            else:
                filtered_dets.append(dets[i])
                filtered_coeffs.append(coeff)
        return filtered_dets, filtered_coeffs

    def get_csf_one_rdm(self, csf_idx, spinor=False):
        r"""
        Gets the 1-RDM (default to spatial MO basis, and NOT spin MO basis) for the CSF
        :return:
        """
        dets, coeffs = self.get_relevant_dets(self.dets_sq, self.csf_coeffs[:, csf_idx])

        spin_one_rdm = get_mc_one_rdm(dets, coeffs)
        if spinor:
            return spin_one_rdm
        else:
            return get_spatial_one_rdm(spin_one_rdm)

    def get_csf_two_rdm(self, csf_idx, spinor=False):
        r"""
        Gets the 2-RDM (default to spatial MO basis, and NOT spin MO basis) for the CSF
        :param spinor:
        :return:
        """
        dets, coeffs = self.get_relevant_dets(self.dets_sq, self.csf_coeffs[:, csf_idx])
        spin_two_rdm = get_ri_mc_two_rdm(dets, coeffs)
        if spinor:
            return spin_two_rdm
        else:
            return get_spatial_two_rdm(spin_two_rdm)
