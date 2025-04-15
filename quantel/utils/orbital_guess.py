from pyscf.mcscf import avas
import numpy as np
from pyscf import scf, lo, ao2mo
from quantel.utils.csf_utils import optimise_order
import scipy.linalg, h5py

def coeff_from_file(integrals,exchange_matrix,filename="guess.hdf5"):

    # Get the number of open-shell orbitals
    nopen = exchange_matrix.shape[0]
    # Create new high-spin PySCF molecule
    pymol = integrals.molecule()
    nelec = pymol.nalfa() + pymol.nbeta()
    ncore = (nelec - nopen) // 2
    
    # Read the coefficients from the file
    print(f"  Reading initial coefficients from {filename}")
    with h5py.File(filename,'r') as F:
        cguess = F['mo_coeff'][:]
    
    # Localise the active orbitals
    print("  Localising open-shell orbitals")
    cactive = np.copy(cguess[:,ncore:ncore+nopen])
    pm = lo.PM(pymol, cactive, scf.ROHF(pymol))
    pm.pop_method = 'becke'
    cactive = pm.kernel()

    # Get exchange integrals in active orbital space
    print("  Computing localised orbital exchange integrals")
    vdm = np.einsum('pi,qi->ipq',cactive,cactive)
    vJ, vK = integrals.build_multiple_JK(vdm,vdm,nopen,nopen)
    # Transform to MO basis
    K = np.einsum('pmn,mq,nq->pq',vK,cactive,cactive)

    # These are the active exchange integrals in chemists order (pq|rs)
    print("  Optimising order of open-shell orbitals")
    order = optimise_order(K, exchange_matrix)

    # Save initial guess and return
    return cguess

def get_avas(integrals, exchange_matrix):
    # Get the number of open-shell orbitals
    ncas = exchange_matrix.shape[0]
    # Create new high-spin PySCF molecule
    pymol = integrals.molecule().copy(deep=True)
    pymol.ms = 0.5 * ncas

    # Run the high-spin ROHF calculation in PySCF
    mf = scf.ROHF(pymol).newton()
    mf.max_cycle = 500
    mf.verbose = 4
    mf.kernel()
    print(f"  High-spin ROHF energy: {mf.e_tot:.10f}")
    
    # Set AO labels
    ao_labels = ['Fe 3d']
    avas_obj = avas.AVAS(mf,ao_labels,canonicalize=True)
    avas_obj.kernel()
    if(avas_obj.nelecas != ncas):
        raise RuntimeError("  Number of AVAS electrons does not match number of CSF orbitals")

    nelec = pymol.nalfa() + pymol.nbeta()
    ncore = (nelec - avas_obj.nelecas) // 2
    ncas = avas_obj.ncas

    cguess  = avas_obj.mo_coeff.copy()
    cactive = cguess[:,ncore:ncore+ncas]
    
    # Localise the active orbitals
    print("  Localising open-shell orbitals")
    pm = lo.PM(pymol, cactive, mf)
    pm.pop_method = 'becke'
    cactive = pm.kernel()

    # Get exchange integrals in active orbital space
    print("  Computing localised orbital exchange integrals")
    vdm = np.einsum('pi,qi->ipq',cactive,cactive)
    vJ, vK = integrals.build_multiple_JK(vdm,vdm,nact,nact)
    # Transform to MO basis
    K = np.einsum('pmn,mq,nq->pq',vK,cactive,cactive)

    # These are the active exchange integrals in chemists order (pq|rs)
    print("  Optimising order of open-shell orbitals")
    order = optimise_order(K, exchange_matrix)

    # Save initial guess and return
    cguess = mf.mo_coeff.copy()
    cguess[:,mf.mo_occ==1] = cactive[:,order]
    return cguess

def rohf_local_guess(integrals, exchange_matrix):
    """
    Get initial orbital guess for CSF method using high-spin ROHF 
    orbitals that are localised to minimise the exchange energy
        Input:
            integrals : Integrals object
            exchange_matrix : Exchange coupling matrix for the CSF
        Returns:
            Cguess    : Initial orbital guess
    """
    # Get the number of open-shell orbitals
    ncas = exchange_matrix.shape[0]
    # Create new high-spin PySCF molecule
    pymol = integrals.molecule().copy(deep=True)
    pymol.ms = 0.5 * ncas

    # Run the high-spin ROHF calculation in PySCF
    mf = scf.ROHF(pymol).newton()
    mf.verbose=4
    mf.max_cycle = 500
    mf.kernel()
    print(f"  High-spin ROHF energy: {mf.e_tot:.10f}")

    # Localise the open-shell orbitals using Pipek-Mezey
    print("  Localising open-shell orbitals")
    cactive = mf.mo_coeff[:,mf.mo_occ==1]
    pm = lo.PM(pymol, cactive, mf)
    pm.pop_method = 'becke'
    cactive = pm.kernel()
    nact = cactive.shape[1]

    # Get exchange integrals in active orbital space
    print("  Computing localised orbital exchange integrals")
    vdm = np.einsum('pi,qi->ipq',cactive,cactive)
    vJ, vK = integrals.build_multiple_JK(vdm,vdm,nact,nact)
    # Transform to MO basis
    K = np.einsum('pmn,mq,nq->pq',vK,cactive,cactive)

    # These are the active exchange integrals in chemists order (pq|rs)
    print("  Optimising order of open-shell orbitals")
    order = optimise_order(K, exchange_matrix)

    # Save initial guess and return
    cguess = mf.mo_coeff.copy()
    cguess[:,mf.mo_occ==1] = cactive[:,order]
    return cguess

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
