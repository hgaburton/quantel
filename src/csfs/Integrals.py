"""
This module deals with the evaluation of 1e and 2e integrals.
Grids used are Treutler-Ahlrichs radial grids and Lebedev
angular grids.
This is taken from the Thom group's holoDFT codes.
"""

from typing import Optional
import numpy as np  # type: ignore
import pyscf  # type: ignore

from pyscf import gto


def get_1e_kinetic(mol: gto.Mole, bas: str):
    r"""Returns the kinetic energy integral.

    Parameters:
        mol: describes the molecule. It contains the basis functions, atoms
            and environment.
        bas: specifies the set of basis functions: ``"cart"`` for Cartesian,
            ``"sph"`` for spherical and ``"spinor"`` for spin-orbital functions.

    Returns:
        A two-dimensional :class:`np.ndarray` storing the required one-electron
        kinetic integrals.
    """
    if bas == "cart":
        ke_int = get_1e_int_gen("int1e_kin_cart", mol, herm=1)
    elif bas == "sph":
        ke_int = get_1e_int_gen("int1e_kin_sph", mol, herm=1)
    elif bas == "spinor":
        ke_int = get_1e_int_gen("int1e_kin", mol, herm=1)
    else:
        raise ValueError("Please specify basis as cart, sph or spinor.")
    return ke_int


def get_1e_int_gen(inttype: str,
                   mol: gto.Mole,
                   dim: Optional[int] = None,
                   herm: int = 0) -> np.ndarray:
    r"""Returns general one-electron integrals.

    Parameters:
        inttype: specifies the type of integral to be evaluated (must match
            ``PySCF`` ``intor_name`` with the basis function specified (as
            ``intor_name_sph`` for spherical basis functions and
            ``intor_name_cart`` for Cartesians. For example, use
            ``"int1e_ovlp_sph"`` for overlap integrals using spherical basis.
            If no basis is specified (eg ``"int1e_ovlp"``), a spinorbital basis
            is assumed.
        mol: describes the molecule. It contains the basis functions, atoms
            and environment.
        dim: (optional) specifies the number of components in the integral
        herm: (optional) specifies the symmetry of the one-electron integrals:

            - ``0`` if no symmetry (default),
            - ``1`` if hermitian, or
            - ``2`` if anti-hermitian.

    Returns:
        A two-dimensional :class:`np.ndarray` storing the required one-electron
        integrals.
    """
    one_e_int = pyscf.gto.moleintor.getints(inttype,
                                            mol._atm,
                                            mol._bas,
                                            mol._env,
                                            shls_slice=None,
                                            comp=dim,
                                            hermi=herm,
                                            aosym=None,
                                            out=None)
    return one_e_int


def get_1e_core(mol: gto.Mole, bas: str) -> np.ndarray:
    r"""Returns the core one-electron Hamiltonian integrals.

    Parameters:
        mol: describes the molecule. It contains the basis functions, atoms
            and environment.
        bas: specifies the set of basis functions: ``"cart"`` for Cartesian,
            ``"sph"`` for spherical and ``"spinor"`` for spin-orbital functions.

    Returns:
        A two-dimensional :class:`np.ndarray` storing the required one-electron
        core Hamiltonian integrals.
    """
    if bas == "cart":
        kinetic = get_1e_kinetic(mol, "cart")
        nuclear = get_1e_int_gen("int1e_nuc_cart", mol, herm=1)
    elif bas == "sph":
        kinetic = get_1e_kinetic(mol, "sph")
        nuclear = get_1e_int_gen("int1e_nuc_sph", mol, herm=1)
    elif bas == "spinor":
        kinetic = get_1e_kinetic(mol, "spinor")
        nuclear = get_1e_int_gen("int1e_nuc", mol, herm=1)
    else:
        raise ValueError("Please specify basis as cart, sph or spinor.")
    assert kinetic.shape == nuclear.shape, \
        "Kinetic and nuclear arrays do not have the same shape."
    one_e_core_int = kinetic + nuclear
    return one_e_core_int


def get_2e_int_gen(inttype: str,
                   mol: gto.Mole,
                   dim: Optional[int] = None,
                   sym: str = '1') -> np.ndarray:
    r"""Returns general two-electron integrals in chemists notation.

    Parameters:
        inttype: specifies the type of integral to be evaluated (must match
            ``PySCF`` ``intor_name`` with the basis function specified (as
            ``intor_name_sph`` for spherical basis functions and
            ``intor_name_cart`` for Cartesians. For example, use
            ``"int2e_sph"`` for two-electron integrals using spherical basis.
            If no basis is specified (eg ``"int2e"``), a spinorbital basis
            is assumed.
        mol: describes the molecule. It contains the basis functions, atoms
            and environment.
        dim: (optional) specifies the number of components in the integral.
        sym: (optional) specifies the symmetry of the two-electron integrals
            :math:`(ij \mid \hat{O} \mid kl)`:

            - ``"1"`` if no symmetry (default),
            - ``"2ij"`` if :math:`i,j`-symmetric,
            - ``"2kl"`` if :math:`k,l`-symmetric, or
            - ``"4"`` if 4-fold symmetry.

    Returns:
        A four-dimensional :class:`np.ndarray` storing the required two-electron
        integrals.
    """
    # pylint: disable=W0212
    # W0212: Accessing protected members of the mol object is required.
    two_e_int = pyscf.gto.moleintor.getints(inttype,
                                            mol._atm,
                                            mol._bas,
                                            mol._env,
                                            comp=dim,
                                            aosym=sym)
    return two_e_int


def get_2e(mol: gto.Mole, bas: str) -> np.ndarray:
    r"""Returns the two-electron integral in chemists' notation,
    :math:`(ij|kl)`.

    Parameters:
        mol: describes the molecule. It contains the basis functions, atoms
            and environment.
        bas: specifies the set of basis functions: ``"cart"`` for Cartesian,
            ``"sph"`` for spherical and ``"spinor"`` for spin-orbital functions.

    Returns:
        A four-dimensional :class:`np.ndarray` storing the required two-electron
        integrals.
    """
    if bas == "cart":
        twoeint = get_2e_int_gen("int2e_cart", mol)
    elif bas == "sph":
        twoeint = get_2e_int_gen("int2e_sph", mol)
    elif bas == "spinor":
        twoeint = get_2e_int_gen("int2e", mol)
    else:
        raise ValueError("Please specify basis as cart, sph or spinor.")
    return twoeint
