import itertools
import numpy as np
from quantel.utils.guga import e_ijji


# This ic just computing the C coefficients, first need to ensure is an allowed combination
# Allowed as Pn can only take an integer value that runs Tn -> -Tn ( its just the projection)  
# Then need to split into the different formulas for tn= 0.5 and =-0.5 
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

#This computes the product of the coefficients that determine the overlap between determinant and CSF 
# isnt used in this or in the csf wfn script so not sure what its useful for?
def get_total_coupling_coefficient(det, csf):
    r"""
    Gets the overlap between the determinant and the CSF. This is the coefficient d of the determinant in the CSF.
    :param det:
    :param csf:
    :return:
    """
    total_coeff = 1 # starting it off from the last term, i.e. t_0 = 1/2, T_0=1/2, P_n=p_n term in coeff = |P_n|^(2) (always has to be) => last coupling coeff is always = 1 
    # product of all the coefficients.
    assert len(det) == len(csf), "Number of orbitals in determinant and CSF are not the same. Check again."#number of orbital in determinant is same as number of electrons. 
    for i in range(1, len(det)):#so now going from t_1, upwards
        Tn = csf[i]
        Pn = det[i]
        tn = csf[i] - csf[i - 1] #using expression that t_n = T_n -T_(n-1)
        pn = det[i] - det[i - 1] # same as above
        total_coeff = total_coeff * get_coupling_coefficient(Tn, Pn, tn, pn) #product sum it 
    return total_coeff

def get_Tn(occstr):
    """ Get the total spin vector for a given occupation string
            :param occstr:
            :return: tn, Tn
    """
    _tn = np.array([(0.5 if(st=='+') else -0.5) for st in occstr])
    _Tn = np.cumsum(_tn)
    return _tn, _Tn
# yh fine this is just converting our +, - into a numerical form.
# occstring is a string of '+-' that refer to the total spin.  


# what is the difference between this and the get_total_coupling_coefficient? only this one is used..., and we have this additional phase factor!
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
    # this phase factor is coming from the number of '-' shells, why should there be this phase factor be here 

    return coeff * phase

# so for a list of determinants, we expect to return a list of coefficients 
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

    if(spin_coupling[0]!='+'): #because t_0=T_0 = 1/2 single electron picture 
        raise RuntimeError("Invalid spin coupling pattern")

    # Get the CSF vectors
    tn, Tn = get_Tn(spin_coupling)# this just gives us the numerical values 

    na = np.sum([s=='+' for s in spin_coupling]) # so for M=S and a given number of electrons, this will define the total spin - two linear equations na+nb = n, na-nb=2M two unknowns.
    for occa in itertools.combinations(range(n),na):
        #itertools.combinations generates all the n-Choose-na combinations of n_a indices. 
        # to generate all the possible determinants with n unpaired, na alpha unpaired, nb beta unpaired we got from M=T_N  
        # Set occupation vector
        occ = np.zeros(n)
        occ[list(occa)] = 1 # all the alpha indices get 1, beta remains 0 
        # Get determinant string
        det = ''.join(['+' if oi==1 else '-' for oi in occ]) # generate the p_n representation of the determinant 
        # Get coefficient
        coeff = get_determinant_coefficient(det, tn, Tn) #so we use this to get the coefficienct, this has that phase term in it - that im not sure we need it? 
        yield det.replace('+','a').replace('-','b'), coeff #so now returns an alpha and beta representation of the determinant, with a the coeff = phase*coeff
#yield stores this in a funny way, we can only access this in an ordered way with a= get_csf_vector(); next(a) ; next(a) etc accessing each tuple at a time and emptying out the store.- its stored as a generator object (single active iterators)  

def distinct_row_table(spin_coupling):
    """ Get the distinct row table for a given spin coupling pattern
            :param spin_coupling:
            :return:
    """
    # what is the distinct row table? 
    return np.array([1 if s == '+' else 2 for s in spin_coupling])

def b_vector(spin_coupling):
    """ Get the b vector for a given spin coupling pattern
            :param spin_coupling:
            :return:
    """
    delta_b = [1 if s == '+' else -1 for s in spin_coupling]
    return np.cumsum(delta_b)


# these are the coefficients from the decomposed one and two electron densities in the generalised Fock matrix elements  
# aij is a matrix in MO basis, 
def get_vector_coupling(nmo, ncore, nocc, spin_coupling):
    """ Compute the vector coupling for active orbitals
            Computes the alpha and beta coefficients for open-shell states
    """
    # Get the number of active orbitals
    nact = nocc - ncore

    # Coulomb coupling matrix
    aij = np.zeros((nmo,nmo))
    aij[:ncore,:ncore] = 4 #puts a 4 in each element for all the core orbitals. - this is like the usual Fock matrix terms in the RHF case 
    aij[:ncore,ncore:nocc] = 2 # core - active 
    aij[ncore:nocc,:ncore] = 2
    aij[ncore:nocc,ncore:nocc] = 1 # active - active 

    # Exchange coupling matrix
    bij = np.zeros((nmo,nmo))
    bij[:ncore,:ncore] = -2 #core-core, why only 2?
    bij[:ncore,ncore:nocc] = -1 #active - core orbitals exchange
    bij[ncore:nocc,:ncore] = -1 # core -active orbitals exchange 
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
