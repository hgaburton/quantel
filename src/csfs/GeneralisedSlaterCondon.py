r"""
This module evaluates Hamiltonian and Overlap matrix elements based on the generalised Slater-Condon rules.
We take arbitrary determinants and find the matrix elements.
This is taken from Thom group's revQCMagic codes.
"""

import numpy as np
from typing import Sequence


def lowdin_pairing(cw: np.ndarray, cx: np.ndarray, sao: np.ndarray,
                   thresh_offdiag: float = 1e-6, thresh_zeroov: float = 1e-6):
    r"""
    Perform Lowdin pairing on a pair of cofficient matrices
    """
    assert (
            cw.shape == cx.shape
    ), f"Coefficient dimensions mismatched: cw{cw.shape} !~ cx{cx.shape}."

    init_orb_Smat = np.einsum(
        "ji,jk,kl->il", cw.conj(), sao, cx, optimize="optimal"
    )

    max_offdiag = np.amax(np.abs(init_orb_Smat - np.diag(np.diag(init_orb_Smat))))

    if max_offdiag <= thresh_offdiag:
        lowdin_overlaps = list(np.diag(init_orb_Smat))
        zero_indices = [
            i for i, ov in enumerate(lowdin_overlaps) if abs(ov) < thresh_zeroov
        ]
        return cw, cx, lowdin_overlaps, zero_indices

    U, _, Vh = np.linalg.svd(init_orb_Smat)
    V = Vh.conj().T
    detV = np.linalg.det(V)

    cwt = np.dot(cw, U)
    detU = np.linalg.det(U)
    cwt[:, 0] *= np.conj(detU)
    cxt = np.dot(cx, V)
    cxt[:, 0] *= np.conj(detV)

    lowdin_orb_Smat = np.linalg.multi_dot([cwt.conj().T, sao, cxt])

    # Checking for diagonality after Löwdin pairing
    max_offdiag = np.amax(np.abs(lowdin_orb_Smat - np.diag(np.diag(lowdin_orb_Smat))))

    assert max_offdiag <= thresh_offdiag, (
            "Löwdin overlap matrix deviates from diagonality. "
            + f"Maximum off-diagonal overlap has magnitude {max_offdiag:.3e} "
            + f"> threshold of {thresh_offdiag:.3e}. Löwdin pairing has failed."
    )

    lowdin_overlaps = list(np.diag(lowdin_orb_Smat))
    zero_indices = [
        i for i, ov in enumerate(lowdin_overlaps) if abs(ov) < thresh_zeroov
    ]
    return cwt, cxt, lowdin_overlaps, zero_indices


def get_overlap(cw, cx, sao):
    r"""
    To a Lowdin overlap matrix [s_1, s_2, ..., s_n], convert all
    """
    mo_overlap_matrix = np.einsum("mn,mi,nj->ij", sao, cw.conj(), cx, optimize="optimal")
    return np.linalg.det(mo_overlap_matrix)


def get_unweighted_codensity_matrix(cwi, cxi):
    r"""
    The unweighted co-density matrix is given by
    {wx}^P_{i} = {w}^C_{i} {x}^C_{i}
    """
    return np.einsum("m,n->mn", cwi, cxi.conj())


def get_weighted_codensity_matrix(cwt, cxt, lowdin_overlaps, zero_indices):
    assert cwt.shape == cxt.shape
    nbasis, nsigma = cwt.shape
    assert len(lowdin_overlaps) == nsigma
    maxdtype = max(
        (ct.dtype for ct in [cwt, cxt]),
        key=lambda dt: dt.num,
        default=np.dtype(np.float64),
    )
    W = np.zeros((nbasis, nbasis), dtype=maxdtype)
    i_nonzero = [i for i in range(nsigma) if i not in zero_indices]
    for i in i_nonzero:
        W = (
                W
                + get_unweighted_codensity_matrix(cwt[:, i], cxt[:, i])
                / lowdin_overlaps[i]
        )
    return W


def get_hamiltonian_element_no_zeros(cwts: Sequence[np.ndarray], cxts: Sequence[np.ndarray],
                                     lowdin_overlapss: Sequence[Sequence[float]], hcore, rij_matrix, e_nuc):
    r"""
    With no zeros, we can use weighted codensity matrices for all the calculations
    """
    nmats = len(cwts)
    wcodens = []
    for spin_idx in range(nmats):
        lowdin_overlaps = lowdin_overlapss[spin_idx]
        wcodens.append(get_weighted_codensity_matrix(cxts[spin_idx], cwts[spin_idx], lowdin_overlaps, []))
    wcodens = np.array(wcodens)
    onee_h = np.einsum("ij,ji", sum(wcodens), hcore)
    twoe_h = np.einsum("ijkl, lk, ji", rij_matrix, sum(wcodens), sum(wcodens),
                       optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, wcodens[0],
                       wcodens[0], optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, wcodens[1],
                       wcodens[1], optimize='optimal')

    return onee_h + 0.5 * twoe_h + e_nuc


def get_hamiltonian_element_one_zeros(cwts: Sequence[np.ndarray], cxts: Sequence[np.ndarray],
                                      lowdin_overlapss: Sequence[Sequence[float]],
                                      zero_indicess: Sequence[Sequence[int]], hcore, rij_matrix):
    r"""
    With one zeros, we have one, two and three electron terms
    """
    nmats = len(cwts)
    n_dim = cwts[0].shape[0]
    wcodens = []
    uwcodens = []
    for spin_idx in range(nmats):
        lowdin_overlaps = lowdin_overlapss[spin_idx]
        zero_indices = zero_indicess[spin_idx]
        wcodens.append(get_weighted_codensity_matrix(cxts[spin_idx], cwts[spin_idx], lowdin_overlaps, zero_indices))
        if len(zero_indices) == 1:
            uwcodens.append(
                get_unweighted_codensity_matrix(cxts[spin_idx][:, zero_indices[0]],
                                                cwts[spin_idx][:, zero_indices[0]])
            )
        else:
            uwcodens.append(np.zeros((n_dim, n_dim)))
    onee_h = sum(np.einsum("ij,ji", uwcoden, hcore) for uwcoden in uwcodens)
    twoe_h = np.einsum("ijkl, lk, ji", rij_matrix, sum(wcodens), sum(uwcodens),
                           optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, wcodens[0],
                       uwcodens[0], optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, wcodens[1],
                       uwcodens[1], optimize='optimal')
    return onee_h + twoe_h


def get_hamiltonian_element_two_zeros(cwts: Sequence[np.ndarray], cxts: Sequence[np.ndarray],
                                      lowdin_overlapss: Sequence[Sequence[float]],
                                      zero_indicess: Sequence[Sequence[int]], rij_matrix):
    r"""
    With two zeros, we need weighted and unweighted codensity matrices
    """
    nmats = len(cwts)
    n_dim = cwts[0].shape[0]
    wcodens = []
    uwcodens = [[], []]
    izeroov = -1
    for spin_idx in range(nmats):
        lowdin_overlaps = lowdin_overlapss[spin_idx]
        zero_indices = zero_indicess[spin_idx]
        wcodens.append(get_weighted_codensity_matrix(cxts[spin_idx], cwts[spin_idx], lowdin_overlaps, zero_indices))
        if len(zero_indices) == 2:
            for zero_index in zero_indices:
                izeroov += 1
                assert izeroov <= 1
                uwcodens[izeroov].append(
                    get_unweighted_codensity_matrix(
                        cxts[spin_idx][:, zero_index], cwts[spin_idx][:, zero_index])
                )
        elif len(zero_indices) == 1:
            izeroov += 1
            assert izeroov <= 1
            uwcodens[izeroov].append(
                get_unweighted_codensity_matrix(
                    cxts[spin_idx][:, zero_indices[0]],
                    cwts[spin_idx][:, zero_indices[0]])
            )
            uwcodens[not izeroov].append(np.zeros((n_dim, n_dim)))
        else:
            uwcodens[0].append(np.zeros((n_dim, n_dim)))
            uwcodens[1].append(np.zeros((n_dim, n_dim)))

    uwcodens = np.array(uwcodens)

    twoe_h = np.einsum("ijkl, lk, ji", rij_matrix, sum(uwcodens[1]), sum(uwcodens[0]),
                       optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, uwcodens[1][0],
                       uwcodens[0][0], optimize='optimal') - \
             np.einsum("ijkl, jk, li", rij_matrix, uwcodens[1][1],
                       uwcodens[0][1], optimize='optimal')
    return twoe_h


def get_det_overlap(state_a, state_b, sao):
    r"""
    Given two coefficient matrices Cwi and Cxi, produce overlap
    :param state_a:
    :param state_b:
    :param sao:
    :return:
    """
    overlap = 1.0
    for i in range(2):  # For each spin-space
        # Perform Lowdin pairing
        cwt, cxt, lowdin_overlaps, zero_indices = lowdin_pairing(state_a[i], state_b[i], sao)
        # Get overlap by multiplying non zero overlaps
        overlap = overlap * get_overlap(cwt, cxt, sao)
    return overlap


def get_matrix_elements(state_a, state_b, sao, hcore, rij_matrix, e_nuc):
    r"""
    Given two coefficient matrices Cwi and Cxi, produce the Hamiltonian element and
    the overlap elements
    """
    cwts = []
    cxts = []
    lowdin_overlapss = []
    zero_indicess = []
    overlap = 1.0
    for i in range(2):  # For each spin-space
        # Perform Lowdin pairing
        cwt, cxt, lowdin_overlaps, zero_indices = lowdin_pairing(state_a[i], state_b[i], sao)
        # Get overlap by multiplying non zero overlaps
        overlap = overlap * get_overlap(cwt, cxt, sao)
        cwts.append(cwt)
        cxts.append(cxt)
        lowdin_overlapss.append(lowdin_overlaps)
        zero_indicess.append(zero_indices)

    # Get reduced overlap
    nmats = len(lowdin_overlapss)
    reduced_ovs = [
        np.prod([ov for i, ov in enumerate(lowdin_overlapss[imat])
                    if i not in zero_indicess[imat]])
        for imat in range(nmats)]
    reduced_ov = np.prod(reduced_ovs)
    nzeros = sum([len(zero_indices) for zero_indices in zero_indicess])
    # Get corresponding hamiltonian element based on number of zeros
    if nzeros == 0:
        hamil = get_hamiltonian_element_no_zeros(cwts, cxts, lowdin_overlapss, hcore, rij_matrix, e_nuc) * reduced_ov
    elif nzeros == 1:
        hamil = get_hamiltonian_element_one_zeros(cwts, cxts, lowdin_overlapss, zero_indicess, hcore, rij_matrix) *\
                reduced_ov
    elif nzeros == 2:
        hamil = get_hamiltonian_element_two_zeros(cwts, cxts, lowdin_overlapss, zero_indicess, rij_matrix) * reduced_ov
    else:
        hamil = 0
    return overlap, hamil
