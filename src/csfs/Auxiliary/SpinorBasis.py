r"""
This module deals with Spinor bases
"""
import numpy as np
from scipy import linalg
from pyscf import gto, scf


def permute_spinor_basis(mat):
    n_dim = len(mat.shape)
    nbsf = mat.shape[0]
    even_perms = [i for i in range(nbsf) if (i % 2 == 0)]
    odd_perms = [i for i in range(nbsf) if (i % 2 == 1)]
    perms = even_perms + odd_perms
    if n_dim == 2:
        return mat[perms, :][:, perms]
    if n_dim == 4:
        return mat[perms, :, :, :][:, perms, :, :][:, :, perms, :][:, :, :, perms]


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


def spinor_one_and_two_e_int(mol, mo_coeff):
    r"""
    Produce one- and two- electron integrals in the spinor basis
    :param mol:
    :param mo_basis:
    :return:
    """
    kin = gto.moleintor.getints("int1e_kin", mol._atm, mol._bas, mol._env, hermi=1)
    vnuc = gto.moleintor.getints("int1e_nuc", mol._atm, mol._bas, mol._env, hermi=1)
    hcore = kin + vnuc
    eri = gto.moleintor.getints("int2e", mol._atm, mol._bas, mol._env)

    # We rearrange the spinor integrals
    spin_hcore = permute_spinor_basis(hcore)
    spin_eri = permute_spinor_basis(eri)

    # We contract AO integrals to give MO integrals
    aug_mo = spatial_to_spin_orbs(mo_coeff, mo_coeff)
    spin_hcore_mo = np.einsum("ip,ij,jq->pq", aug_mo, spin_hcore, aug_mo)
    spin_eri_mo = np.einsum("ip,jq,ijkl,kr,ls->pqrs", aug_mo, aug_mo, spin_eri, aug_mo, aug_mo)
    return spin_hcore_mo, spin_eri_mo
