r"""
RDMapper expresses RDM objects in a way that works with PySCF
The RDMs are also constructed from PySCF's FCI module
"""

import numpy as np
from itertools import combinations


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

def make_rdm12(civec, norbs, nelec, spin, link_index=None, reorder=True):
    r"""
    PySCF's RDM makers assume that number of alpha and beta electrons differ by at most 1.
    Technically _unpack_nelec can take into account spins but none of the RDM makers do.
    This method thus generalises the functionality given by fci.direct_spin1.make_rdm12
    """
    import ctypes
    from pyscf import lib
    from pyscf.fci import cistring, rdm
    from pyscf.fci.addons import _unpack_nelec
    librdm = lib.load_library('libfci')

    fname = 'FCIrdm12kern_sf'
    symm = 0
    assert civec is not None
    cibra = np.asarray(civec, order='C')
    ciket = np.asarray(civec, order='C')
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norbs), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norbs), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert (cibra.size == na*nb)
    assert (ciket.size == na*nb)
    rdm1 = np.empty((norbs,norbs))
    rdm2 = np.empty((norbs,norbs,norbs,norbs))
    librdm.FCIrdm12_drv(getattr(librdm, fname),
                        rdm1.ctypes.data_as(ctypes.c_void_p),
                        rdm2.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norbs),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
    dm1 = rdm1.T
    dm2 = rdm2
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def get_dm12(kets, coeffs, spin):
    civec, norbs, nelec = mapper(kets, coeffs)
    dm1, dm2 = make_rdm12(civec, norbs, nelec, spin)
    return dm1, dm2
