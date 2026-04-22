#!/usr/bin/python

import numpy as np
from quantel import FCIDumpInterface

class BaseMolecule:
    """Simple molecule class to hold electron counts for a FCIDUMP."""
    def __init__(self,nalfa,nbeta):
        self._nalfa = nalfa
        self._nbeta = nbeta
    
    def nalfa(self):
        return self._nalfa
    
    def nbeta(self):
        return self._nbeta

    def nelec(self):
        return self._nalfa + self._nbeta

    def sz(self):
        return 0.5 * (self._nalfa - self._nbeta)

class FCIDUMP:
    """Integral interface backed by a pre-computed FCIDUMP file.

    The orbitals in the FCIDUMP are assumed to be orthonormal, so the
    overlap and orthogonalisation matrices are both the identity.

    This class exposes the same API as PySCFIntegrals so that it can be
    used as a drop-in replacement wherever only a FCIDUMP is available.

    Parameters
    ----------
    filename : str
        Path to the FCIDUMP file.
    """

    def __init__(self, filename: str):
        """Initialize the FCIDUMP interface by reading the specified file.
        Parameters
        ----------
        filename : str
            Path to the FCIDUMP file.
        """
        self.filename = filename
        self._ints = FCIDumpInterface(filename)
        self.mol = BaseMolecule(self._ints.nalfa(), self._ints.nbeta())
        self.xc = None
        self.hybrid_K = 1.0

    def molecule(self) -> BaseMolecule:
        """Return a molecule object with the correct number of electrons."""
        return self.mol
    
    def nbsf(self) -> int:
        """Number of orbitals (NORB from FCIDUMP header)."""
        return self._ints.nbsf()

    def nmo(self) -> int:
        """Number of molecular orbitals (equal to nbsf for an orthogonal FCIDUMP)."""
        return self._ints.nmo()

    def scalar_potential(self) -> float:
        """Scalar potential stored in the FCIDUMP (nuclear repulsion or frozen-core energy)."""
        return self._ints.scalar_potential()

    def overlap_matrix(self) -> np.ndarray:
        """Return the overlap matrix (identity, since orbitals are orthonormal)."""
        return self._ints.overlap_matrix()

    def orthogonalization_matrix(self) -> np.ndarray:
        """Return the orthogonalisation matrix (identity, since orbitals are orthonormal)."""
        return self._ints.orthogonalization_matrix()

    def oei_matrix(self, spin=None) -> np.ndarray:
        """Return the one-electron Hamiltonian matrix h(p,q).

        Parameters
        ----------
        spin : bool or None
            Ignored — alpha and beta integrals are identical for a restricted FCIDUMP.
        """
        return self._ints.oei_matrix(True)

    def dipole_matrix(self, origin=None):
        """Not available from a FCIDUMP file."""
        raise RuntimeError(
            "INTDUMPMolecule: dipole integrals are not available from a FCIDUMP file"
        )

    def build_fock(self, dens: np.ndarray) -> np.ndarray:
        """Build the restricted Fock matrix F = h + (2J-K) from a density matrix.

        Parameters
        ----------
        dens : ndarray, shape (nbsf, nbsf)
            Density matrix in the orbital basis.

        Returns
        -------
        ndarray, shape (nbsf, nbsf)
            Fock matrix.
        """
        return self._ints.build_fock(np.asarray(dens, dtype=np.float64))
    
    def build_vxc(self, dens):
        # Return 0 for the XC potential since a FCIDUMP contains no information about an XC functional.
        return 0, np.zeros_like(dens)

    def build_JK(self,vDJ,vDK,hermi=0,Kxc=False):
        """Build J and K matrices for multiple sets of density matrices.

        Parameters
        ----------
        vDJ : ndarray, shape (nj, nbsf, nbsf)
            Density matrices for J build.
        vDK : ndarray, shape (nk, nbsf, nbsf)
            Density matrices for K build.
        nj, nk : int
            Number of density matrices in each set.

        Returns
        -------
        tuple of ndarray
            (vJ, vK) each shaped (n*, nbsf, nbsf).
        """
        if(vDJ.ndim == 2): vDJ = vDJ[None,:,:]
        if(vDK.ndim == 2): vDK = vDK[None,:,:]
        # Make sure inputs are C-contiguous arrays of type float64
        vDJ = np.ascontiguousarray(vDJ, dtype=np.float64)
        vDK = np.ascontiguousarray(vDK, dtype=np.float64)
        # Call the underlying C++ implementation to compute J and K for all density matrices
        vJ, vK = self._ints.build_multiple_JK(vDJ, vDK, vDJ.shape[0], vDK.shape[0])
        vKfunc = vK
        if Kxc:
            return vJ, vK, vKfunc
        else:
            return vJ, vK


    def oei_ao_to_mo(self, C1: np.ndarray, C2: np.ndarray, spin=None) -> np.ndarray:
        """Transform the one-electron integrals to a new orbital basis.

        Computes C1.T @ h @ C2 where h is the stored one-electron matrix.

        Parameters
        ----------
        C1, C2 : ndarray, shape (nbsf, n*)
            Orbital coefficient matrices.
        spin : bool or None
            Ignored — alpha and beta integrals are identical.

        Returns
        -------
        ndarray, shape (n1, n2)
        """
        return self._ints.oei_ao_to_mo(np.asarray(C1, dtype=np.float64),
                                       np.asarray(C2, dtype=np.float64),True)

    def tei_ao_to_mo(self,C1: np.ndarray,C2: np.ndarray,C3: np.ndarray,C4: np.ndarray,
                     alpha1: bool,alpha2: bool,) -> np.ndarray:
        """Transform the two-electron integrals to a new orbital basis.

        Returns antisymmetrised integrals <pq||rs> in physicist's notation.
        Same-spin (alpha1 == alpha2) integrals are antisymmetrised; opposite-
        spin integrals are not.

        Parameters
        ----------
        C1 : ndarray, shape (nbsf, n1)
            Orbital coefficients for electron 1 bra state.
        C2 : ndarray, shape (nbsf, n2)
            Orbital coefficients for electron 2 bra state.
        C3 : ndarray, shape (nbsf, n3)
            Orbital coefficients for electron 1 ket state.
        C4 : ndarray, shape (nbsf, n4)
            Orbital coefficients for electron 2 ket state.
        alpha 1 : bool
            Spin of electron 1. [True=alpha|False=beta]
        alpha 2 : bool
            Spin of electron 2. [True=alpha|False=beta]

        Returns
        -------
        ndarray, shape (n1, n2, n3, n4)
        """
        return self._ints.tei_ao_to_mo(
            np.asarray(C1, dtype=np.float64),
            np.asarray(C2, dtype=np.float64),
            np.asarray(C3, dtype=np.float64),
            np.asarray(C4, dtype=np.float64),
            alpha1, alpha2,
        )

    def tei_array(self, spin1=None, spin2=None) -> np.ndarray:
        """Return the full two-electron integral array (pq|rs) in chemist's notation.

        Parameters
        ----------
        spin1, spin2 : ignored
            Included for API compatibility with PySCFIntegrals.

        Returns
        -------
        ndarray, shape (nbsf, nbsf, nbsf, nbsf)
        """
        return self._ints.tei_array()

    def mo_integrals(self, C: np.ndarray, ncore: int = 0, nactive: int = 0):
        """Build a MOintegrals object for the (ncore, nactive) partition of C.

        Parameters
        ----------
        C : ndarray, shape (nbsf, nmo)
            Full orbital coefficient matrix.
        ncore : int
            Number of core (inactive) orbitals.
        nactive : int
            Number of active orbitals (defaults to nmo - ncore).

        Returns
        -------
        MOintegrals
        """
        return self._ints.mo_integrals(np.asarray(C, dtype=np.float64), ncore, nactive)

    def print(self):
        """Print a summary of the FCIDUMP contents."""
        print(f"  FCIDUMP file : {self.filename}")
        print(f"  NORB         : {self.nbsf()}")
        print(f"  NELEC        : {self.molecule().nelec()}  "
              f"(nalpha={self.molecule().nalfa()}, nbeta={self.molecule().nbeta()})")
        print(f"  Scalar pot.  : {self.scalar_potential(): .10f}")

    def __str__(self):
        return (
            f"<quantel.ints.fcidump_integrals.FCIDUMP: "
            f"norb={self.nbsf()}, nelec={self.molecule().nelec()}, file='{self.filename}'>"
        )