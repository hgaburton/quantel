from abc import ABCMeta
import numpy as np
from ReducedDensityMatrices.RDMapper import mapper, get_dm12


class GenericCSF(metaclass=ABCMeta):

    def __init__(self, stot: float, ms: float, n_core: int, n_act: int, n_elec: int):
        self._stot = stot
        self._ms = ms
        self._n_core = n_core
        self._n_act = n_act
        self._n_elec = n_elec


    @property
    def stot(self):
        """Spin quantum number"""
        return self._stot

    @property
    def ms(self):
        """Spin projection value"""
        return self._ms

    @property
    def n_core(self):
        """Number of core orbitals"""
        return self._n_core

    @property
    def n_act(self):
        """Number of active orbitals"""
        return self._n_act

    @property
    def n_elec(self):
        """Number of electrons"""
        return self._n_elec

    @property
    def n_alpha(self):
        """Number of alpha electrons"""
        return int(self.n_elec + 2 * self.ms) // 2

    @property
    def n_beta(self):
        """Number of beta electrons"""
        return int(self.n_elec - 2 * self.ms) // 2

    def dets_sq(self):
        """Determinants in Second Quantisation"""
        pass

    def csf_coeffs(self):
        """Coefficient on each determinant"""
        pass

    @staticmethod
    def get_relevant_dets(dets, coeffs, thresh=1e-10):
        """Filters a list of determinants (in Second Quantised representation)
            based on the coefficients. If coefficients are smaller than the given threshold, ignore"""
        filtered_dets = []
        filtered_coeffs = []
        for i, coeff in enumerate(coeffs):
            if np.isclose(coeff, 0, rtol=0, atol=thresh):
                pass
            else:
                filtered_dets.append(dets[i])
                filtered_coeffs.append(coeff)
        return filtered_dets, filtered_coeffs

    def get_pyscf_rdms(self):
        """Gets the spatial 1-RDM and 2-RDM from PySCF"""
        dets, coeffs = self.get_relevant_dets(self.dets_sq, self.csf_coeffs)
        dm1, dm2 = get_dm12(dets, coeffs, self.ms * 2)
        return dm1, dm2

    def get_civec(self):
        """Gets CI vector in PySCF format"""
        dets, coeffs = self.get_relevant_dets(self.dets_sq, self.csf_coeffs)
        civec, norbs, nelec = mapper(dets, coeffs)
        return civec

    def get_s2(self):
        """Gets S^2 value"""
        pass