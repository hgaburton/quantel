import itertools
import numpy as np
from quantel.utils.guga import e_ijji
from quantel.utils.linalg import pseudo_inverse

def get_coupling_coefficient(Tn, Pn, tn, pn):
    """ Computes the coupling coefficient C_{tn, pn}^{Tn, Pn}
            :param Tn:
            :param Pn:
            :param tn:
            :param pn:
            :return:
    """
    # This is a forbidden case
    if Tn < np.abs(Pn):
        return 0
    if np.isclose(0.5, tn, rtol=0, atol=1e-10):
        return np.sqrt((Tn + 2 * pn * Pn) / (2 * Tn))
    elif np.isclose(-0.5, tn, rtol=0, atol=1e-10):
        return -2 * pn * np.sqrt((Tn + 1 - 2 * pn * Pn) / (2 * (Tn + 1)))
    else:
        raise RuntimeError("Invalid spin coupling coefficient requested")

def get_total_coupling_coefficient(det, csf):
    r"""
    Gets the overlap between the determinant and the CSF. This is the coefficient d of the determinant in the CSF.
    :param det:
    :param csf:
    :return:
    """
    total_coeff = 1
    assert len(det) == len(csf), "Number of orbitals in determinant and CSF are not the same. Check again."
    for i in range(1, len(det)):
        Tn = csf[i]
        Pn = det[i]
        tn = csf[i] - csf[i - 1]
        pn = det[i] - det[i - 1]
        total_coeff = total_coeff * get_coupling_coefficient(Tn, Pn, tn, pn)
    return total_coeff

def get_Tn(occstr):
    """ Get the total spin vector for a given occupation string
            :param occstr:
            :return: tn, Tn
    """
    _tn = np.array([(0.5 if(st=='+') else -0.5) for st in occstr])
    _Tn = np.cumsum(_tn)
    return _tn, _Tn


def get_determinant_coefficient(det, tn, Tn):
    """ Get the coefficient for a given determinant
            :param det:
            :param tn:
            :param Tn:
            :return:
    """
    # Get the total spin vector for the determinant
    pn, Pn = get_Tn(det)
    
    # Get the coupling coefficient
    coeff = np.prod([get_coupling_coefficient(Tn[i], Pn[i], tn[i], pn[i]) for i in range(len(tn))])

    # Get the phase
    phase = 1
    for j, st1 in enumerate(det):
        for st2 in det[j+1:]:
            if(st1=='-' and st2=='+'):
                phase *= -1

    return coeff * phase

def get_csf_vector(spin_coupling):
    """ Iterate over the list of determinants and their coefficients for a given CSF
            :param csf:
            :return:
    """   
    # Check CSF vector is valid
    n  = len(spin_coupling)
    if(n==0):
        yield '', 1
        return

    if(spin_coupling[0]!='+'):
        raise RuntimeError("Invalid spin coupling pattern")

    # Get the CSF vectors
    tn, Tn = get_Tn(spin_coupling)

    na = np.sum([s=='+' for s in spin_coupling])
    for occa in itertools.combinations(range(n),na):
        # Set occupation vector
        occ = np.zeros(n)
        occ[list(occa)] = 1
        # Get determinant string
        det = ''.join(['+' if oi==1 else '-' for oi in occ])
        # Get coefficient
        coeff = get_determinant_coefficient(det, tn, Tn)
        yield det.replace('+','a').replace('-','b'), coeff

def distinct_row_table(spin_coupling):
    """ Get the distinct row table for a given spin coupling pattern
            :param spin_coupling:
            :return:
    """
    return np.array([1 if s == '+' else 2 for s in spin_coupling])

def b_vector(spin_coupling):
    """ Get the b vector for a given spin coupling pattern
            :param spin_coupling:
            :return:
    """
    delta_b = [1 if s == '+' else -1 for s in spin_coupling]
    return np.cumsum(delta_b)

def get_vector_coupling(nmo, ncore, nocc, spin_coupling):
    """ Compute the vector coupling for active orbitals
            Computes the alpha and beta coefficients for open-shell states
    """
    # Get the number of active orbitals
    nact = nocc - ncore

    # Coulomb coupling matrix
    aij = np.zeros((nmo,nmo))
    aij[:ncore,:ncore] = 4
    aij[:ncore,ncore:nocc] = 2
    aij[ncore:nocc,:ncore] = 2
    aij[ncore:nocc,ncore:nocc] = 1

    # Exchange coupling matrix
    bij = np.zeros((nmo,nmo))
    bij[:ncore,:ncore] = -2
    bij[:ncore,ncore:nocc] = -1
    bij[ncore:nocc,:ncore] = -1
    # Contributions from GUGA
    drt = distinct_row_table(spin_coupling)
    bvec = b_vector(spin_coupling)
    for i in range(nact):
        for j in range(nact):
            bij[i+ncore,j+ncore] = e_ijji(bvec,drt,i,j)

    return aij, bij

def get_shell_exchange(ncore, shell_indices, spin_coupling):
    """ Compute the exchange contributions for each unique shell pair"""
    # Get the distinct row table and b vector
    drt = distinct_row_table(spin_coupling)
    bvec = b_vector(spin_coupling)
    
    # Get the number of shells
    nshell = len(shell_indices)
    # Compute beta matrix
    beta = np.zeros((nshell,nshell))
    for w in range(nshell):
        for v in range(w,nshell):
            beta[w,v] = e_ijji(bvec,drt,shell_indices[w][0]-ncore,shell_indices[v][0]-ncore)
            beta[v,w] = beta[w,v]
    return beta

def get_shells(ncore, spin_coupling):
    """ Get the indices of orbitals within shells
            :param ncore:
            :param spin_coupling:
            :return:
    """
    # Initialise with core (doubly occupied) shell
    core_indices = list(range(ncore))
    shell_indices = []
    if(len(spin_coupling)>0):
        # Get indices for each shell
        active_shells = np.cumsum([0 if i==0 else spin_coupling[i-1]!=spin_coupling[i] 
                                for i in range(len(spin_coupling))])
        for i in range(active_shells[-1]+1):
            shell_indices.append((ncore+np.argwhere(active_shells==i).ravel()).tolist())
    return core_indices, shell_indices

def get_det_occupation(shell_spin, shell_indices, ncore, nmo):
    """ Get the occupation vector from a specified shell spin pattern
            :param shell_spin:      Spin pattern per shell 
            :param shell_indices:   Indices of orbitals per shell
            :param ncore:           Number of core orbitals
            :param nmo:             Total number of molecular orbitals
            :return: occa, occb
    """
    # Initialise occupation vectors
    occa, occb = np.zeros(nmo), np.zeros(nmo)
    # Core orbitals
    occa[:ncore] = 1
    occb[:ncore] = 1
    # Open-shell orbitals
    for i, shell in enumerate(shell_indices):
        occa[shell] = 1 if (shell_spin[i] == 'a') else 0
        occb[shell] = 0 if (shell_spin[i] == 'a') else 1
    return occa, occb

def optimise_order(K, X):
    """
    Discrete (local) optimisation of the open-shell orbital ordering through orbital swaps. 
    This algorithm is designed to quickly minimise the exchange energy, but is not guaranteed 
    to find the global minimum.
        Input:
            K : Matrix of exchange integrals in the active orbital space <pq|qp>
            X : Matrix of exchange coupling terms Xpq
        Returns:
            order : Optimised orbital ordering
    """
    norb = K.shape[0]
    order = np.arange(norb)
    for it in range(10):
        sweep_swap = False
        for i in range(norb):
            for j in range(i):
                Knew = K.copy()
                Knew[[i,j],:] = Knew[[j,i],:]
                Knew[:,[i,j]] = Knew[:,[j,i]]
                if(0.5 * np.vdot(Knew-K, X) < -1e-10):
                    # Swap the orbitals
                    order[[j,i]] = order[[i,j]]
                    K[[i,j],:] = K[[j,i],:]
                    K[:,[i,j]] = K[:,[j,i]]
                    sweep_swap = True

        # If we have not made any further swaps, we are done.
        if(not sweep_swap):
            break

    return order

def csf_reorder_orbitals(integrals, exchange_matrix, cinit, pop_method='becke'):  
    """
    Optimise the order of the CSF orbitals using the exchange matrix 
    to minimise the exchange energy
        Input:
            integrals : Integrals object
            exchange_matrix : Exchange coupling matrix for the CSF
            cinit    : Initial open-shell orbitals
        Returns:
            copt     : Optimised orbital guess
    """
    from pyscf import scf, lo

    # Get the number of open-shell orbitals
    nopen = cinit.shape[1]
    if(exchange_matrix.shape[0] != nopen):
        raise RuntimeError("  Number of CSF orbitals does not match number of open-shell orbitals")

    # Access the PySCF molecule object
    pymol = integrals.molecule()

    # Localise the active orbitals
    print("  Localising open-shell orbitals")
    pm = lo.PM(pymol, cinit, scf.ROHF(pymol))
    pm.pop_method = pop_method
    cinit = pm.kernel()

    # Get exchange integrals in active orbital space
    print("  Computing localised orbital exchange integrals")
    vdm = np.einsum('pi,qi->ipq',cinit,cinit)
    vJ, vK = integrals.build_multiple_JK(vdm,vdm,nopen,nopen)
    # Transform to MO basis
    K = np.einsum('pmn,mq,nq->pq',vK,cinit,cinit)

    # These are the active exchange integrals in chemists order (pq|rs)
    print("  Optimising order of open-shell orbitals")
    order = optimise_order(K, exchange_matrix)

    # Save initial guess and return
    copt = cinit[:,order].copy()
    return copt

def csf_det_list(spin_coupling):
    """ Get the list of open-shell determinants for a given CSF pattern
            :param spin_coupling:
            :return:
    """
    # Check CSF vector is valid
    n = len(spin_coupling)
    if(n==0):
        return [""]

    if(spin_coupling[0]!='+'):
        raise RuntimeError("Invalid spin coupling pattern")
    
    na = np.sum([s=='+' for s in spin_coupling])
    det_list = []
    for occa in itertools.combinations(range(n),na):
        # Set occupation vector
        occ = np.zeros(n)
        occ[list(occa)] = 1
        # Get determinant string
        det = ''.join(['a' if oi==1 else 'b' for oi in occ])
        det_list.append(det)
    return det_list

def get_uhf_coupling(occstr):
    """ Get the UHF/UKS exchange coupling matrix for a given occupation string
            :param occstr:
            :return:
    """
    nopen = len(occstr)
    Kmat = np.zeros((nopen,nopen))
    for p,pchar in enumerate(occstr):
        for q, qchar in enumerate(occstr):
            if(pchar==qchar):
                Kmat[p,q] = -1
    return Kmat

def get_ensemble_expansion(spin_coupling):
    """ Get the ensemble expansion of energy in terms of UHF/UKS determinant energies
        We achieve this by matching the exchange matrices of CSF and ensemble

            :param spin_coupling:
            :return: list of (determinant string, coefficient) tuples
    """
    if(len(spin_coupling)==0 or spin_coupling=='cs'):
        return []
    # Get length of spin coupling pattern
    nopen = len(spin_coupling)
    ncore = 0 
    # Get indices of shells
    shells = get_shells(ncore,spin_coupling)[1]
    # Get number of shells
    nshell = len(shells)
    # Define function to vectorize shell matrices
    vector_inds = (np.insert(np.tril_indices(nshell,k=-1)[0],0,0),
                      np.insert(np.tril_indices(nshell,k=-1)[1],0,0))
    vec = lambda M : M[vector_inds]
    # Get Beta vector
    v_beta = vec(get_shell_exchange(ncore,shells,spin_coupling))

    # Now we construct the determinants with at most 2 shell flips
    # Here, we denote only the spin per shell
    dets = []
    if(nshell==4 or nshell<=2):
        for it in itertools.combinations(range(0,nshell),r=1):
            spins = np.zeros(nshell)
            # Flip the spin of a shell
            spins[it[0]]=1
            # Make sure first shell is always alpha
            if(spins[0] == 1): spins = 1 - spins
            # Get string format
            spin_str = ''.join(['a' if s==0 else 'b' for s in spins])
            # Add to dets if not present
            if(not spin_str in [d[0] for d in dets]): 
                dets.append((spin_str, vec(get_uhf_coupling(spin_str))))
    else:
        spin_str = 'a'*nshell
        dets.append((spin_str, vec(get_uhf_coupling(spin_str))))

    for it in itertools.combinations(range(0,nshell),r=2):
        spins = np.zeros(nshell)
        # Flip the spins of two shells
        spins[it[0]]=1
        spins[it[1]]=1
        # Make sure first shell is always alpha
        if(spins[0] == 1): spins = 1 - spins
        # Get string format
        spin_str = ''.join(['a' if s==0 else 'b' for s in spins])
        # Add to dets if not present
        if(not spin_str in [d[0] for d in dets]): 
            dets.append((spin_str, vec(get_uhf_coupling(spin_str))))
    # Sort lexicographically for clarity
    dets.sort()

    # We now have a basis to expand our exchange matrix in. 
    # We need to buld the metric tensor and solve for the expansion coefficients
    nbasis = len(dets)
    metric = np.zeros((nbasis,nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            metric[i,j] = np.dot(dets[i][1], dets[j][1])
    # Build pseudo-inverse
    X = np.linalg.pinv(metric)

    # Form the projection of target exchange matrix
    bvec = np.array([np.dot(dets[i][1],v_beta) for i in range(nbasis)])
    coeffs = X @ bvec

    # Form the expansion and return
    expansion = [(idet[0],coeff) for idet, coeff in zip(dets, coeffs)]
    #print("Exchange matrix expansion coefficients:")
    #for detI, cI in expansion:
    #    print(f"{detI}: {cI: 8.4f}")
    return expansion
