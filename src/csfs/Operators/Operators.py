r"""
Operators are assembled here
"""
import copy
import numpy as np


def phase_factor(p, ket):
    total_filled_before = np.sum(ket[1:p])
    return (-1) ** total_filled_before


def create(p, ket):
    r"""
    Creation operator for the p-th spin orbital
    :param p:
    :param ket:
    :return:
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    else:
        if ket[p] == 1:
            return 0
        else:
            pf = phase_factor(p, ket)
            ket[p] = 1
            ket[0] *= pf
            return ket


def annihilate(p, ket):
    r"""
    Annihilation operator for the p-th spin orbital
    :param p:
    :param ket:
    :return:
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    else:
        if ket[p] == 0:
            return 0
        else:
            pf = phase_factor(p, ket)
            ket[p] = 0
            ket[0] *= pf
            return ket


def excitation(p, q, ket):
    r"""
    The one-electron excitation operator a^{\dagger}_{p} a_{q}
    :param p:
    :param q:
    :param ket:
    :return:
    """
    return create(p, annihilate(q, ket))


def overlap(bra, ket):
    r"""
    Find overlap between the bra and the ket
    :param bra:
    :param ket:
    :return:
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    if type(bra) == int:
        if bra == 0:
            return 0
    if all(x == y for x, y in zip(bra[1:], ket[1:])):
        return 1
    else:
        return 0


def overlap_diffbas(bra, ket, cross_overlap_mat):
    r"""
    Finds the overlap between bra and ket where the underlying MO representations
    are DIFFERENT
    :param bra:
    :param ket:
    """
    if type(ket) == int:
        if ket == 0:
            return 0
    if type(bra) == int:
        if bra == 0:
            return 0
    arr_bra = copy.deepcopy(np.array(bra[1:]))
    arr_ket = copy.deepcopy(np.array(ket[1:]))
    return np.linalg.det(np.einsum("p,pq,q->pq", arr_bra, cross_overlap_mat, arr_ket))
