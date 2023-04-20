r"""
RDMapper expresses RDM objects in a way that works with PySCF
The RDMs are also constructed from PySCF's FCI module
"""

import numpy as np
from itertools import permutations, combinations
from pyscf import fci


def get_occupation_dicts(norbs, nelec):
    r"""
    Given number of spatial orbitals and number of electrons in a spin space, produce
    a dictionary with the bitwise representation of electron occupation as keys e.g. 1100, 0011.
    These keys correspond to a value (integer, zero ordering) which indicates their position on
    a coefficient matrix.

    For 2 electrons in 4 orbitals, the key/ value pairs will look like:
    1100: 0
    1010: 1
    0110: 2
    1001: 3
    0101: 4
    0011: 5
    """
    occ_reprs = list(combinations(np.arange(norbs), nelec))
    occ_dict = {}
    for i, occ_repr in enumerate(occ_reprs):
        temp = np.zeros(norbs, dtype=int)
        for _, idx in enumerate(occ_repr):
            temp[idx] = 1
        occ_str = ''.join(str(x) for x in temp)
        occ_dict[occ_str] = i
    return occ_dict


def mapper(kets, coeffs):
    r"""
    Given a list of kets and their coefficients, produce FCI formatted matrix 
    """
    assert len(kets) == len(coeffs)
    nspatialorbs = (len(kets[0]) - 1) // 2
    nalpha = int(np.sum(kets[0][1:nspatialorbs+1]))
    nbeta = int(np.sum(kets[0][nspatialorbs+1:]))
    alpha_occ_dict = get_occupation_dicts(nspatialorbs, nalpha)
    beta_occ_dict = get_occupation_dicts(nspatialorbs, nbeta)
    mat = np.zeros((len(alpha_occ_dict), len(beta_occ_dict)))
    for i, ket in enumerate(kets):
        pf = ket[0]
        alpha_str = ''.join(str(x) for x in ket[1:nspatialorbs+1])
        beta_str = ''.join(str(x) for x in ket[nspatialorbs+1:])
        mat[alpha_occ_dict.get(alpha_str)][beta_occ_dict.get(beta_str)] += coeffs[i] * pf
    return mat, nspatialorbs, (nalpha, nbeta)

def get_dm12(kets, coeffs):
    civec, norbs, nelec = mapper(kets, coeffs)
    dm1, dm2 = fci.direct_spin1.make_rdm12(civec, norbs, nelec)
    return dm1, dm2
