r"""
We construct reduced density matrices (RDMs) in this module.
This uses a rather inefficient implementation (can be further worked on in the future)

This assumes orthonormal orbitals.
"""

import numpy as np
import itertools


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


def build_1rdm(o_coeffs, v_coeffs):
    r"""
    Builds 1RDM
    :param coeffs:
    :return:
    """
    o_dim = o_coeffs.shape[1]
    v_dim = v_coeffs.shape[1]
    rdm1 = np.zeros((o_dim + v_dim, o_dim + v_dim))
    for i in range(o_dim):
        rdm1[i][i] = 1
    return rdm1


def build_1etd(n_dim, bra_idxs, ket_idxs):
    r"""
    Builds 1 electron transition density matrices

    :return:
    """
    etd1 = np.zeros((n_dim, n_dim))
    common_so_idxs = list(set(bra_idxs).intersection(set(ket_idxs)))
    bra_diff_idxs = list(set(bra_idxs) - set(common_so_idxs))
    ket_diff_idxs = list(set(ket_idxs) - set(common_so_idxs))
    if len(bra_diff_idxs) == 0:
        print("This should not be possible for a transition density matrix")
    elif len(bra_diff_idxs) > 1:
        return etd1  # All zeros as the operator is a one electron term
    else:
        # The term is only non zero if it matches the orbital that is excited
        p = bra_diff_idxs[0]
        r = ket_diff_idxs[0]
        etd1[p][r] = 1
        return etd1


def o_build_2rdm(o_coeffs, v_coeffs):
    r"""
    Builds 2RDM
    :param o_coeffs:
    :param v_coeffs:
    :return:
    """
    o_dim = o_coeffs.shape[1]
    v_dim = v_coeffs.shape[1]
    n_dim = o_dim + v_dim
    # Build 1RDM
    rdm1 = build_1rdm(o_coeffs, v_coeffs)
    v_perm = list(itertools.combinations(np.arange(n_dim), o_dim))
    # Build 1e TDM
    etd1s = []
    for idx, perm in enumerate(v_perm[1:]):
        etd1 = build_1etd(n_dim, np.arange(o_dim), list(perm))
        etd1s.append(etd1)
    etd1s = np.array(etd1s)
    # Bring it all together
    rdm2 = np.einsum("pr,qs->pqrs", rdm1, rdm1) + \
           np.einsum("vpr,vsq->pqrs", etd1s, etd1s) -\
           np.einsum("qr,ps->pqrs", rdm1, np.identity(n_dim))
    return rdm2


def get_2edm(rdm2):
    r"""
    Gets the 2 electron density matrix. The trace of this matrix should equal 0.5 * N * (N-1) where N is the number of
    electrons (number of e pairs)
    :param rdm2:
    :return:
    """
    n_dim = rdm2.shape[0]
    mat = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(i + 1):
            mat[i][j] += rdm2[i][j][i][j]
            mat[j][i] += rdm2[i][j][i][j]
    return mat


def get_partial_trace(rdm2):
    r"""
    Takes the partial trace of the 2RDM
    :param rdm2:
    :return:
    """
    n_dim = rdm2.shape[0]
    sum = 0
    for i in range(n_dim):
        for j in range(i + 1):
            sum += rdm2[i][j][i][j]
    return sum
