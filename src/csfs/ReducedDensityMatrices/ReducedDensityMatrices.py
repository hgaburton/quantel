r"""
We form the 1-RDM and 2-RDM for single- and multi-configurational wavefunctions
"""
import copy
import numpy as np
import itertools
from Operators.Operators import create, annihilate, excitation, overlap


def get_ri_kets(ket):
    r"""
    Gets the full list of kets for RI calculations
    :param ket:
    :return: A list of lists
    """
    ri_kets = []
    n_orbs = len(ket) - 1
    n_alpha = int(np.sum(ket[1:n_orbs // 2 + 1]))
    n_beta = int(np.sum(ket[n_orbs // 2 + 1:]))
    alpha_rep = list(itertools.combinations(np.arange(n_orbs // 2), n_alpha))
    beta_rep = list(itertools.combinations(np.arange(n_orbs // 2), n_beta))
    reps = list(itertools.product(alpha_rep, beta_rep))
    for i, rep in enumerate(reps):
        ket = np.zeros(len(ket))
        for a in rep[0]:  # Alpha indices
            ket[a + 1] = 1
        for b in rep[1]:
            ket[1 + n_orbs//2 + b] = 1
        ket[0] = 1
        ri_kets.append(ket)
    return ri_kets


def get_one_rdm(bra, ket):
    r"""
    Obtain the 1-RDM corresponding to the bra and ket given
    < A | a^{\dagger}_{p} a_{q} | B >
    :param bra:
    :param ket:
    :return:
    """
    n_dim = len(ket) - 1  # Number of MOs
    one_rdm = np.zeros((n_dim, n_dim))
    for p in range(n_dim):
        for q in range(n_dim):
            # 1-ordering here
            mod_ket = excitation(p + 1, q + 1, copy.deepcopy(ket))
            if type(mod_ket) == int or type(bra) == int: #  The only int value it can take is zero
                assert (mod_ket == 0 or bra == 0) #  Just to be safe, we assert it
                one_rdm[p][q] = 0
            else:
                one_rdm[p][q] = overlap(bra, mod_ket) * mod_ket[0] * bra[0]
    return one_rdm


def get_two_rdm(bra, ket):
    r"""
    Obtain the 2-RDM using a brute-force approach
    < A | a^{\dagger}_{p} a^{\dagger}_{r} a_{s} a_{q} | B >
    :param bra:
    :param ket:
    :return:
    """
    n_dim = len(ket) - 1  # Number of MOs
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    if type(bra) == int or type(ket) == int:  # No computation required if bra or ket is zero
        assert (bra == 0 or ket == 0)
        return two_rdm
    for p in range(n_dim):
        for q in range(n_dim):
            for r in range(n_dim):
                for s in range(n_dim):
                    mod_ket = create(p + 1,
                                     create(r + 1,
                                            annihilate(s + 1,
                                                       annihilate(q + 1,
                                                                  copy.deepcopy(ket)))))
                    if type(mod_ket) == int:
                        assert mod_ket == 0
                        two_rdm[p][q][r][s] = 0
                    else:
                        two_rdm[p][q][r][s] = overlap(bra, mod_ket) * mod_ket[0] * bra[0]
    return two_rdm


def get_mc_one_rdm(kets, coeffs):
    r"""
    Obtain the 1-RDM corresponding to a multi-configurational wavefunction
    < A | a^{\dagger}_{p} a_{q} | B >
    :param kets:
    :param coeffs:
    :return:
    """
    n_dim = len(kets[0]) - 1  # Number of MOs
    n_kets = len(kets)  # Number of states
    assert n_kets == len(coeffs)  # Each ket must have a coefficient
    one_rdm = np.zeros((n_dim, n_dim))
    for i in range(n_kets):
        for j in range(n_kets):
            one_rdm += get_one_rdm(kets[i], kets[j]) * coeffs[i] * coeffs[j]
    return one_rdm


def get_mc_two_rdm(kets, coeffs):
    r"""
    Obtain the 2-RDM corresponding to a multi-configurational wavefunction
    < A | a^{\dagger}_{p} a^{\dagger}_{r} a_{s} a_{q} | B >
    :param kets:
    :param coeffs:
    :return:
    """
    n_dim = len(kets[0]) - 1  # Number of MOs
    n_kets = len(kets)  # Number of states
    assert n_kets == len(coeffs)  # Each ket must have a coefficient
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for i in range(n_kets):
        for j in range(n_kets):
            two_rdm += get_two_rdm(kets[i], kets[j]) * coeffs[i] * coeffs[j]
    return two_rdm


def get_spatial_one_rdm(spin_1rdm):
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


def get_spatial_two_rdm(spin_2rdm):
    r"""
    Change the RDM from spin orbital basis to spatial orbital basis by doing a spin summation
    :param spatial_2rdm:
    :return: 2-RDM in spatial orbital basis
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


def get_partial_trace(rdm):
    r"""
    Given a 2-RDM, we get the partial trace. This should correspond to the number
    of electron pairs.
    """
    trace = 0
    n_dim = rdm.shape[0]
    for i in range(n_dim):
        for j in range(n_dim):
            trace += rdm[i][i][j][j]
    return trace * 0.5


def get_ri_two_rdm(bra, ket):
    r"""
    Obtain the 2-RDM by exploiting the resolution of the identity
    :param ket:
    :return:
    """
    n_dim = len(ket) - 1  # Number of MOs
    ri_kets = get_ri_kets(ket)
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for k, ri_ket in enumerate(ri_kets):
        bra_one_rdm = get_one_rdm(copy.deepcopy(bra), copy.deepcopy(ri_ket))
        ket_one_rdm = get_one_rdm(copy.deepcopy(ri_ket), copy.deepcopy(ket))
        two_rdm += np.einsum("pq,rs->pqrs", bra_one_rdm, ket_one_rdm)
    two_rdm -= np.einsum("ps,qr->pqrs", get_one_rdm(bra, ket), np.identity(n_dim))
    return two_rdm


def get_ri_mc_two_rdm(kets, coeffs):
    r"""
    Obtain the 2-RDM corresponding to a multi-configurational wavefunction
    < A | a^{\dagger}_{p} a^{\dagger}_{r} a_{s} a_{q} | B >
    by exploiting the resolution of the identity
    :param kets:
    :param coeffs:
    :return:
    """
    n_dim = len(kets[0]) - 1  # Number of MOs
    n_kets = len(kets)  # Number of states
    assert n_kets == len(coeffs)  # Each ket must have a coefficient
    two_rdm = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for i in range(n_kets):
        for j in range(n_kets):
            two_rdm += get_ri_two_rdm(copy.deepcopy(kets[i]), copy.deepcopy(kets[j])) * coeffs[i] * coeffs[j]
    return two_rdm
