r"""
This module deals with Spinor bases
"""
import numpy as np
from scipy import linalg
from pyscf import gto, scf


def spatial_one_and_two_e_int(mol, mo_coeff):
    r"""
    Produce one- and two- electron integrals in the spatial basis
    :param mol:
    :param mo_basis:
    :return:
    """
    kin = mol.intor('int1e_kin')
    vnuc = mol.intor('int1e_nuc')
    hcore = kin + vnuc
    eri = mol.intor('int2e')

    hcore_mo = np.einsum("ip,ij,jq->pq", mo_coeff, hcore, mo_coeff)
    eri_mo = np.einsum("ip,jq,ijkl,kr,ls->pqrs", mo_coeff, mo_coeff, eri, mo_coeff, mo_coeff)
    return hcore_mo, eri_mo
