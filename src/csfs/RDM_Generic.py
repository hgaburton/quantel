r"""
This module computes the RDM for a generic wavefunction which is expressed as a
linear combination of Slater Determinants.

The Slater Determinants are allowed to be non-orthogonal to each other. The algorithm works by
performing bi-orthogonalisation/ Lowdin pairing of any two pairs of Slater determinants.
The relevant 1RDM and 2RDM are formed for each pair.

The 1RDM and 2RDM of the wavefunctions can then be constructed with this information.
"""
import numpy as np

from itertools import combinations
from csfs.CSFConstructor import CSFConstructor
from csfs.GeneralisedSlaterCondon import get_det_overlap

np.set_printoptions(precision=6, suppress=True)


def augment_to_so(alpha_coeffs, beta_coeffs, alpha_occ_idxs, beta_occ_idxs):
    r"""
    Augments coefficients to spin orbital format
    :param alpha_coeffs:
    :param beta_coeffs:
    :return:
    """
    nbsf = alpha_coeffs.shape[0]
    nmos = alpha_coeffs.shape[1]
    alpha_vir_idxs = list(set(np.arange(nmos)) - set(alpha_occ_idxs))
    beta_vir_idxs = list(set(np.arange(nmos)) - set(beta_occ_idxs))
    return np.hstack([np.vstack([alpha_coeffs[:, alpha_occ_idxs], np.zeros((nbsf, len(alpha_occ_idxs)))]),
                      np.vstack([np.zeros((nbsf, len(beta_occ_idxs))), beta_coeffs[:, beta_occ_idxs]]),
                      np.vstack([alpha_coeffs[:, alpha_vir_idxs], np.zeros((nbsf, len(alpha_vir_idxs)))]),
                      np.vstack([np.zeros((nbsf, len(beta_vir_idxs))), beta_coeffs[:, beta_vir_idxs]]),
                      ])


def spatial_to_spin_orbs(alpha_coeffs, beta_coeffs):
    r"""
    If one has alpha and beta coeffiicent matrices, convert it into spin orbital format
    :param alpha_coeffs:
    :param beta_coeffs:
    :return:
    """
    nbsf = alpha_coeffs.shape[0]
    assert alpha_coeffs.shape[0] == beta_coeffs.shape[0], "Different AO basis set size used for alpha and beta spaces"
    a_nmos = alpha_coeffs.shape[1]
    b_nmos = beta_coeffs.shape[1]
    return np.hstack([np.vstack([alpha_coeffs, np.zeros((nbsf, a_nmos))]),
                      np.vstack([np.zeros((nbsf, b_nmos)), beta_coeffs])
                      ])


def augment_sao(sao):
    r"""
    Augments the SAO matrix into the spin-orbital basis
    :param sao:
    :return:
    """
    return np.hstack([np.vstack([sao, np.zeros(sao.shape)]), np.vstack([np.zeros(sao.shape), sao])])


def get_det_index(csfobj: CSFConstructor, state):
    r"""
    For a state which is described by a determinant, find the its index in the CSFConstructor object
    :param csfobj:
    :param state:
    :return:
    """
    rep = tuple([tuple(state[0]), tuple(state[1])])
    return csfobj.dets_orbrep.index(rep)


def build_1rdm(csfobj: CSFConstructor, n_dim, ref_state_idx: int, unchanged_idx: int = 0):
    r"""
    Build 1RDM naively
    Unchanged_idx corresponds to index of the bra state
    Ref_state_idx corresponds to index of the ket state
    :param unchanged_idx:
    :param csfobj:
    :param n_dim:
    :param ref_state_idx: :int: The index corresponding to the determinant of the reference state, \Psi_{0}
    :return:
    """
    rdm1 = np.zeros((n_dim, n_dim))
    for r in range(n_dim):  # This is the creation index
        for p in range(n_dim):  # This is the annihilation index
            if r < (n_dim // 2) <= p:  # If creation and annihilation have different spins, it is zero
                rdm1[r][p] = 0
            elif r >= (n_dim // 2) > p:  # If creation and annihilation have different spins, it is zero
                rdm1[r][p] = 0
            else:
                alpha_idxs = list(csfobj.dets_orbrep[ref_state_idx][0])
                beta_idxs = list(csfobj.dets_orbrep[ref_state_idx][1])
                state = [alpha_idxs, beta_idxs]
                if p < (n_dim // 2):  # Alpha spin
                    if p in state[0]:
                        state[0].remove(p)
                        state[0].append(r)
                        state[0].sort()
                        # Find the index that this state corresponds to
                        rep = tuple([tuple(state[0]), tuple(state[1])])
                        if rep not in csfobj.dets_orbrep:
                            rdm1[r][p] = 0
                        else:
                            idx = csfobj.dets_orbrep.index(rep)
                            rdm1[r][p] = get_det_overlap(csfobj.get_det(unchanged_idx), csfobj.get_det(idx),
                                                         csfobj.overlap)
                    else:
                        rdm1[r][p] = 0
                else:  # Beta spin
                    p_mod = p - (n_dim // 2)
                    r_mod = r - (n_dim // 2)
                    if p_mod in state[1]:
                        state[1].remove(p_mod)
                        state[1].append(r_mod)
                        state[1].sort()
                        # Find the index that this state corresponds to
                        rep = tuple([tuple(state[0]), tuple(state[1])])
                        if rep not in csfobj.dets_orbrep:
                            rdm1[r][p] = 0
                        else:
                            idx = csfobj.dets_orbrep.index(rep)
                            rdm1[r][p] = get_det_overlap(csfobj.get_det(unchanged_idx), csfobj.get_det(idx),
                                                         csfobj.overlap)
                    else:
                        rdm1[r][p] = 0
    return rdm1


def build_1etd(csfobj: CSFConstructor, n_dim, bra_state_idx, ket_state_idx):
    r"""
    Builds 1 electron transition density matrices
    :return:
    """
    etd1s_bra = []
    etd1s_ket = []
    for v in range(csfobj.n_dets):
        etd1_bra = build_1rdm(csfobj, n_dim, bra_state_idx, v)
        etd1_ket = build_1rdm(csfobj, n_dim, ket_state_idx, v)
        etd1s_bra.append(etd1_bra)
        etd1s_ket.append(etd1_ket)
    return np.array(etd1s_bra), np.array(etd1s_ket)


def build_2rdm(csfobj: CSFConstructor, n_dim, bra_state_idx: int, ket_state_idx: int):
    r"""
    Builds 2RDM
    :param o_coeffs:
    :param v_coeffs:
    :return:
    """
    # Build 1RDM
    rdm1 = build_1rdm(csfobj, n_dim, ket_state_idx, bra_state_idx)
    # Build 1e TDM
    etd1s_bra, etd1s_ket = build_1etd(csfobj, n_dim, bra_state_idx, ket_state_idx)
    # Bring it all together
    rdm2 = np.einsum("vqp,vrs->pqrs", etd1s_bra, etd1s_ket) - \
           np.einsum("ps,qr->pqrs", rdm1, np.identity(n_dim))
    return rdm2


def build_generic_1rdm(csfobj: CSFConstructor, csf_idx):
    r"""
    Builds 1RDM representing a CSF
    :param csfobj: CSFConstructor object
    :param csf_idx: :int: The index of the CSF required
    :return: 2RDM of the CSF
    """
    # To avoid doing needless computation, we filter out CSF coefficients which are zero
    csf_coeff = csfobj.csf_coeffs[:, csf_idx]
    nmos = csfobj.coeffs.shape[1] * 2
    total_1rdm = np.zeros((csfobj.n_dets, csfobj.n_dets, nmos, nmos))
    for det_i in range(csfobj.n_dets):
        for det_j in range(csfobj.n_dets):
            total_1rdm[det_i, det_j, :, :] = build_1rdm(csfobj, nmos, det_i, det_j)
    return total_1rdm


def build_generic_2rdm(csfobj: CSFConstructor, csf_idx):
    r"""
    Builds 2RDM representing a CSF
    :param csfobj: CSFConstructor object
    :param csf_idx: :int: The index of the CSF required
    :return: 2RDM of the CSF
    """
    # To avoid doing needless computation, we filter out CSF coefficients which are zero
    generic_1rdm = build_generic_1rdm(csfobj, csf_idx)
    csf_coeffs = csfobj.csf_coeffs
    nmos = csfobj.coeffs.shape[1] * 2
    total_2rdm = np.einsum("klqp,kmrs,l,m,l,m->pqrs", generic_1rdm, generic_1rdm,
                           csf_coeffs[:, csf_idx], csf_coeffs[:, csf_idx],
                           csfobj.det_phase_factors, csfobj.det_phase_factors) - \
                 np.einsum("lmps,qr,l,m,l,m->pqrs", generic_1rdm, np.identity(nmos),
                           csf_coeffs[:, csf_idx], csf_coeffs[:, csf_idx],
                           csfobj.det_phase_factors, csfobj.det_phase_factors)
    return total_2rdm


def spin_to_spatial_1rdm(spin_1rdm):
    r"""
    Change the RDM from spin orbital basis to spatial orbital basis by doing a spin summation
    :param spatial_1rdm:
    :return: 1-RDM in spatial orbital basis
    """
    n_dim = spin_1rdm.shape[0] // 2     # This should be an integer
    spatial_1rdm = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            spatial_1rdm[i][j] += spin_1rdm[i][j] + spin_1rdm[i+n_dim][j+n_dim]
    return spatial_1rdm


def spin_to_spatial_2rdm(spin_2rdm):
    r"""
    Change the RDM from spin orbital basis to spatial orbital basis by doing a spin summation
    :param spatial_1rdm:
    :return: 1-RDM in spatial orbital basis
    """
    n_dim = spin_2rdm.shape[0] // 2     # This should be an integer
    spatial_2rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for p in range(n_dim):
        for q in range(n_dim):
            for r in range(n_dim):
                for s in range(n_dim):
                    spatial_2rdm[p][q][r][s] += spin_2rdm[p][q][r][s]
                    spatial_2rdm[p][q][r][s] += spin_2rdm[p + n_dim][q + n_dim][r + n_dim][s + n_dim]
                    # p and q have the same spin, r and s have the same spin
                    spatial_2rdm[p][q][r][s] += spin_2rdm[p][q][r + n_dim][s + n_dim]
                    spatial_2rdm[p][q][r][s] += spin_2rdm[p + n_dim][q + n_dim][r][s]
    return spatial_2rdm


def get_rdm12(csfobj: CSFConstructor, csf_idx):
    r"""
    Builds 2RDM representing a CSF
    :param csfobj: CSFConstructor object
    :param csf_idx: :int: The index of the CSF required
    :return: 2RDM of the CSF
    """
    spatial_1rdm = spin_to_spatial_1rdm(build_generic_1rdm(csfobj, csf_idx)[csf_idx][csf_idx])
    spatial_2rdm = spin_to_spatial_2rdm(build_generic_2rdm(csfobj, csf_idx))
    return spatial_1rdm, spatial_2rdm
