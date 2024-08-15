import sys, itertools
import numpy as np
from quantel.utils.guga import e_ijji

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
        return '', 1
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
                                for i in range(len(spin_coupling ))])
        for i in range(active_shells[-1]+1):
            shell_indices.append((ncore+np.argwhere(active_shells==i).ravel()).tolist())
    return core_indices, shell_indices
