from pyscf.mcscf import avas
import numpy as np
from pyscf import scf
import scipy.linalg, h5py


def coeff_from_file(filename="guess.hdf5"):
    """
    Read the initial orbital guess from a file
        Input:
            filename : Name of the file to read
        Returns:
            cguess   : Initial orbital guess
    """
    # Read the coefficients from the file
    print(f"  Reading initial coefficients from {filename}")
    with h5py.File(filename,'r') as F:
        return np.copy(F['mo_coeff'][:])


def get_avas(integrals, ao_labels,ms=0):
    """
    Get initial orbital guess using AVAS method with predetermined AO labels
        Input:
            integrals : Integrals object
            ao_labels : List of atomic orbital labels
        Returns:
            Cguess    : Initial orbital guess
    """
    if(ao_labels is None):
        raise RuntimeError("No atomic orbital labels provided for AVAS method")

    # Create new high-spin PySCF molecule
    pymol = integrals.molecule().copy(deep=True)
    pymol.ms = ms

    # Run the high-spin ROHF calculation in PySCF
    mf = scf.ROHF(pymol).newton()
    mf.max_cycle = 500
    mf.verbose = 4
    mf.kernel()
    print(f"  High-spin ROHF energy: {mf.e_tot:.10f}")
    
    # Set AO labels
    avas_obj = avas.AVAS(mf,ao_labels,canonicalize=True)
    avas_obj.kernel()
    if(avas_obj.nelecas != int(2*ms)):
        raise RuntimeError("  Number of AVAS electrons does not match number of CSF orbitals")

    return avas_obj.mo_coeff.copy()


def rohf_local_guess(integrals,ms=0):
    """
    Get initial orbital guess using high-spin ROHF orbitals
        Input:
            integrals : Integrals object
        Returns:
            Cguess    : Initial orbital guess
    """
    # Create new high-spin PySCF molecule
    pymol = integrals.molecule().copy(deep=True)
    pymol.ms = ms

    # Run the high-spin ROHF calculation in PySCF
    mf = scf.ROHF(pymol).newton()
    mf.verbose=4
    mf.max_cycle = 500
    mf.kernel()
    print(f"  High-spin ROHF energy: {mf.e_tot:.10f}")
    return mf.mo_coeff.copy()


def core_guess(integrals):
    """
    Get initial orbital guess by diagonalising core Hamiltonian
        Input:
            integrals : Integrals object
        Returns:
            Cguess    : Initial orbital guess
    """
    # Get the core Hamiltonian and overlap matrix
    h1e = integrals.oei_matrix(True)
    s = integrals.overlap_matrix()
    return scipy.linalg.eigh(h1e, s)[1]


def gwh_guess(integrals, K=1.75):
    """
    Get initial orbital guess using GWH method
        Input:
            integrals : Integrals object
        Returns:
            Cguess    : Initial orbital guess
    """
    # Get the core Hamiltonian and overlap matrix
    h1e = integrals.oei_matrix(True)
    s = integrals.overlap_matrix()
    nbsf = integrals.nbsf()

    # Build GWH guess Hamiltonian
    hguess = np.zeros((nbsf,nbsf))
    for i in range(nbsf):
        for j in range(nbsf):
            hguess[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
            if(i!=j):
                hguess[i,j] *= 1.75
    # Solve initial generalised eigenvalue problem
    return scipy.linalg.eigh(hguess, s)[1]

def orbital_guess(integrals, method="gwh", avas_ao_labels=None, gwh_K=1.75, rohf_ms=0):
    """
    Get initial orbital guess using different methods
        Input:
            integrals : Integrals object
            method    : Method to use for initial guess
            avas_ao_labels : List of atomic orbital labels (for AVAS method only)
        Returns:
            Cguess    : Initial orbital guess
    """
    if(method.lower() == "core"):
        from quantel.utils.orbital_guess import core_guess
        Cguess = core_guess(integrals)
    elif(method.lower() == "gwh"):
        from quantel.utils.orbital_guess import gwh_guess
        Cguess = gwh_guess(integrals,K=gwh_K)
    elif(method.lower() == "avas"):
        from quantel.utils.orbital_guess import get_avas
        Cguess = get_avas(integrals,avas_ao_labels,ms=rohf_ms)
    elif(method.lower() == "rohf"):
        from quantel.utils.orbital_guess import rohf_local_guess
        Cguess = rohf_local_guess(integrals,ms=rohf_ms)
    elif(method.lower() == "fromfile"):
        from quantel.utils.orbital_guess import coeff_from_file
        Cguess = coeff_from_file()
    else:
        raise NotImplementedError(f"Orbital guess method {method} not implemented")
   
    return Cguess
