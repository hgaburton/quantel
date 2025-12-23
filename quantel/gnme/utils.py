#!/usr/bin/python3 

import unittest
import numpy as np
import pygnme

def orthogonalisation_matrix(M, thresh=1e-8):
    """Construct an orthogonalisation matrix X such that X^T M X = I for a symmetric matrix M
    
        Inputs:
        -------
            M      2d-array containing symmetric matrix to orthogonalise
            thresh (optional) Threshold for determining if eigenvalue is zero
        Outputs:
        --------
            X      2d-array containing orthogonalisation matrix
    """
    # Get eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(M)

    # Identify non-null space
    inds = np.argwhere(np.abs(eigval) > thresh).flatten()

    # Construct orthogonalisation matrix
    X = eigvec[:,inds].dot(np.diag(np.power(eigval[inds],-0.5)))

    return X

def occstring_to_bitset(occstr):
    """Convert an occupation string to a pair of pygnme bitsets"""
    occa = []
    occb = []
    for i in occstr:
        if(i=='2'):
            occa.append(1)
            occb.append(1)
        elif(i=='a'):
            occa.append(1)
            occb.append(0)
        elif(i=='b'):
            occa.append(0)
            occb.append(1)
        elif(i=='0'):
            occa.append(0)
            occb.append(0)
        else:
            raise ValueError('Invalid character in occupation string')
    return pygnme.utils.bitset(list(reversed(occa))), pygnme.utils.bitset(list(reversed(occb)))

def gen_eig_sym(M, S, thresh=1e-8):
    """ Solve the generalised eigenvalue problem Mx = Sx lambda for a symmetric matrix M and S
    
        Inputs:
        -------
            M      2d-array containing symmetric matrix
            S      2d-array containing symmetric matrix
            thresh (optional) Threshold for determining if eigenvalue is zero
        Outputs:
        --------
            eigval 1d-array containing eigenvalues
            eigvec 2d-array containing eigenvectors
    """
    # Check the input
    assert(M.shape[0] == S.shape[0])
    assert(M.shape[1] == S.shape[1])

    # Build orthogonalisation matrix
    X = orthogonalisation_matrix(S, thresh=thresh)

    # Project into non-null subspace
    Mp = np.conj(X).T.dot(M.dot(X))

    # Solve orthogonalised problem
    eigval, eigvec_p = np.linalg.eigh(Mp)

    # Project back into full space
    eigvec = X.dot(eigvec_p)

    return eigval, eigvec

def orthogonalise(C,S,thresh=1e-8):
    """ Orthogonalise a set of occipied orbitals C with respect to the overlap matrix S

        Inputs:
        -------
            C      2d-array containing occupied orbitals
            S      2d-array containing overlap matrix
            thresh (optional) Threshold for determining if overlap is already diagonal
        Outputs:
        --------
            C_new  2d-array containing orthogonalised orbitals
    """

    # Get MO overlap matrix
    Smo = np.einsum('pi,pq,qj->ij',np.conj(C),S,C)

    # Get orthogonalisation matrix
    X = orthogonalisation_matrix(Smo, thresh=thresh)

    # Orthogonalise the orbitals
    C_new = np.einsum('pi,ij->pj',C,X)

    return C_new

def lowdin_pair(Cw,Cx,S,thresh=1e-10):
    '''Perform Lowdin pairing on two sets of *occupied* orbitals defined in Cw (bra) and Cx (ket).

       Inputs:
       -------
           Cw      2d-array containing occupied orbitals in bra state
           Cx      2d-array containing occupied orbitals in ket state
           S       Overlap matrix for corresponding basis orbitals
           thresh  (optional) Threshold for determining if overlap is already diagonal

       Outputs: 
       --------
           Cw_new  2d-array containing Lowdin-paired orbitals in bra state
           Cx_new  2d-array containing Lowdin-paired orbitals in ket state
           Sxx     1d-array containing diagonal elements of biorthogonalised overlap matrix
           
       return Cw_new, Cx_new, Sxx
    '''
    # Get overlap matrix 
    Swx = np.einsum('pi,pq,qj->ij',np.conj(Cw),S,Cx)

    # Take a copy of coefficients for the output
    Cw_new = Cw.copy()
    Cx_new = Cx.copy()

    # No pairing needed if off-diagonal is zero
    diag_test = Swx - np.diag(np.diag(Swx))

    # Otherwise perform the pairing
    if(np.linalg.norm(diag_test) > thresh):
        # Construct transformation matrices using SVD
        U, D, VT = np.linalg.svd(Swx)
        V = np.conj(VT).T

        # Transform the orbitals
        Cw_new = np.einsum('pi,ij->pj',Cw,U)
        Cx_new = np.einsum('pi,ij->pj',Cx,V)

        # Correct for any new phase
        Cw_new[:,0] *= np.linalg.det(np.conj(U).T)
        Cx_new[:,0] *= np.linalg.det(np.conj(V).T)

        # Compute new overlap matrix
        Swx = np.einsum('pi,pq,qj->ij',np.conj(Cw_new),S,Cx_new)

    # Get diagonal of overlap matrix
    Sxx = np.diag(Swx)

    # Return the result
    return Cw_new, Cx_new, Sxx

def factorised_densities(Cwa,Cwb,Cxa,Cxb,metric,stol=1e-8,ktol=1e-8,thresh=1e-8):
    """ 
    Compute the factorised density matrices for nonorthogonal matrix element 
    with up to 2-body coupling.
     
       Inputs:
       -------
           Cax, Cbx     2x 1d-arrays containing alfa and beta coefficients for bra state
           Caw, Cbw     2x 1d-arrays containing alfa and beta coefficients for ket state
           metric       AO overlap matrix
           ktol (optional) Threshold for cutting off kappa singular values.
           stol (optional) Threshold for cutting of zero overlap orbital pairs.
           thresh (optional) Threshold diagonal comparison in Lowdin pairing

       return S, (Wa,Ma,Pa), (Wb,Mb,Pb), kt
    """
    # Perform Lowdin pairing for the alfa and beta orbital spaces
    Cwa_new, Cxa_new, Sa = lowdin_pair(Cwa,Cxa,metric,thresh=thresh)
    Cwb_new, Cxb_new, Sb = lowdin_pair(Cwb,Cxb,metric,thresh=thresh)

    # Compute the epislon tensors
    na, nb = Sa.shape[0], Sb.shape[0]
    Sxx = np.hstack([Sa, Sb])
    norb = na + nb
    nbsf = metric.shape[0]

    # Build single orbital epsilon tensors
    ei  = np.zeros((norb))
    for i in range(norb):
        mask = np.ones(norb, dtype=bool)
        mask[i] = False
        ei[i] = np.prod(Sxx[mask])

    # Build double orbital epsilon tensors
    eij = np.zeros((norb,norb))
    for i in range(norb):
        for j in range(norb):
            mask = np.ones(norb, dtype=bool)
            mask[i] = False
            mask[j] = False
            eij[i,j] = 0 if i==j else np.prod(Sxx[mask])
    
    # Build the W matrix and record lower than treshold values
    Wa, Wb = np.zeros((nbsf,nbsf)), np.zeros((nbsf,nbsf))
    for i, si in enumerate(Sa):
        if(abs(si) > stol):
            Wa += np.outer(Cxa_new[:,i],Cwa_new[:,i]) / si
    for i, si in enumerate(Sb):
        if(abs(si) > stol):
            Wb += np.outer(Cxb_new[:,i],Cwb_new[:,i]) / si

    # Find indices of small overlap terms
    vind = np.argwhere(np.abs(Sxx) < stol).ravel()
    nsmall = len(vind)

    # Build M matrix
    Ma, Mb = np.zeros((nbsf,nbsf)), np.zeros((nbsf,nbsf))
    for v in vind:
        if(v<na):
            # Contributes to alfa space
            Ma += np.outer(Cxa_new[:,v],Cwa_new[:,v]) * ei[v]
        else:
            # Contributes to beta space
            Mb += np.outer(Cxb_new[:,v-na],Cwb_new[:,v-na]) * ei[v]

    # Build P matrices for small terms
    Pva = np.zeros((nsmall,nbsf,nbsf))
    Pvb = np.zeros((nsmall,nbsf,nbsf))
    for i, v in enumerate(vind):
        if(v<na):
            Pva[i] = np.outer(Cxa_new[:,v],Cwa_new[:,v])
        else:
            Pvb[i] = np.outer(Cxb_new[:,v-na],Cwb_new[:,v-na])

    # Diagonalise evw for small overlap terms
    evw = eij[vind,:][:,vind]
    kt, vt = np.linalg.eigh(evw)

    # Transform the P matrices
    Pa = np.einsum('kij,kt->tij',Pva,vt)
    Pb = np.einsum('kij,kt->tij',Pvb,vt)

    return np.prod(Sxx), (Wa,Ma,Pa), (Wb,Mb,Pb), kt

def generalised_slater_condon(Cax,Cbx,Caw,Cbw,ints,ktol=1e-8,stol=1e-8):
    """ 
    Compute a matrix element for two-body operator using generalised Slater-Condon rules
     
       Inputs:
       -------
           Cax, Cbx     2x 1d-arrays containing alfa and beta coefficients for bra state
           Caw, Cbw     2x 1d-arrays containing alfa and beta coefficients for ket state
           ints         Integral object holding relevant AO integrals
           ktol (optional) Threshold for cutting off kappa singular values.
           stol (optional) Threshold for cutting of zero overlap orbital pairs.

       Outputs:
       --------
           S       Overlap of the two determinants
           H       Hamiltonian coupling of the two determinants

       return S, H
    """
    metric = ints.overlap_matrix()
    hcore  = ints.oei_matrix(True)

    S, (Wa,Ma,Pa), (Wb,Mb,Pb), kt = factorised_densities(Cax,Cbx,Caw,Cbw,metric,stol,ktol)
    nd = 2 + len(kt)

    # Overlap term
    H = S * ints.scalar_potential()

    # One-body matrix
    H += S * np.einsum('ij,ji', hcore, Wa + Wb) + np.einsum('ij,ji', hcore, Ma + Mb)

    # JK builds
    vd = np.zeros((2*nd,ints.nbsf(),ints.nbsf()))
    vd[0] = Wa
    vd[1] = Ma
    vd[2:nd] = Pa
    vd[nd] = Wb
    vd[nd+1] = Mb
    vd[nd+2:] = Pb

    # Call the JK build
    vj, vk = ints.build_JK(vd,vd)

    # Compute two-electron contributions
    jw_t = vj[0] + vj[nd]
    Wt = Wa + Wb
    Mt = Ma + Mb
    H += 0.5 * np.einsum('ij,ji',jw_t, S * Wt + 2 * Mt)
    H -= 0.5 * np.einsum('ij,ji',vk[0],  S * Wa + 2 * Ma)
    H -= 0.5 * np.einsum('ij,ji',vk[nd], S * Wb + 2 * Mb)
    for t, ek in enumerate(kt):
        Pk = Pa[t] + Pb[t]
        Jk = vj[2+t] + vj[nd+2+t]

        H += 0.5 * ek * np.einsum('ij,ji',Jk, Pk)
        H -= 0.5 * ek * np.einsum('ij,ji',vk[2+t], Pa[t])
        H -= 0.5 * ek * np.einsum('ij,ji',vk[nd+2+t], Pb[t])

    return S, H

def reduced_overlap(Sxx, thresh=1e-8):
    '''Evaluate the reduced overlap from an array of biorthogonal overlap elements
     
       Inputs:
       -------
           Sxx     1d-array containing diagonal elements of biorthogonalised overlap matrix
           thresh  (optional) Threshold for determining if an overlap term is zero.

       Outputs:
       --------
           invSxx  1d-array containing diagonal elements of inverse biorthogonalised overlap matrix
           redOV   Reduced overlap (product of non-zero overlap terms)
           zeros   Indices of zero overlap terms
           nzero   Number of zero overlap terms

       return invSxx, redOV, zeros, nzero
    '''
    # Prepare inverse overlap
    invSxx = np.ones(Sxx.shape)
    invSxx[np.abs(Sxx)>thresh] = np.power(Sxx[np.abs(Sxx)>thresh],-1)

    # Get the reduced overlap
    redOV = np.prod(Sxx[np.abs(Sxx)>thresh])

    # Get indices of zero overlaps and count them
    zeros = np.argwhere(np.abs(Sxx) < thresh)
    nzero = len(zeros)

    return invSxx, redOV, zeros, nzero


class test_utils(unittest.TestCase):
    def get_random_coeff(self,nb,ne):
        return np.random.rand(nb,ne)

    def get_random_overlap(self,nb):
        # Fill with random numbers
        S  = np.random.rand(nb,nb)
        # Make sure it's symmetric
        S  = 0.5 * (S + S.T)
        S  = S - np.diag(np.diag(S)) + np.eye(nb)
        # Make sure it's positive definite
        sig, U = np.linalg.eigh(S)
        S  = np.einsum('pq,qr,rs->ps',U,np.diag(np.abs(sig)),U.T)
        return S

    def test_lowdin_pair(self):

        # Set some number of electrons (ne) and basis functions (nb)
        ne,nb = 10, 5

        # Get random coefficients
        Cw = self.get_random_coeff(nb,ne)
        Cx = self.get_random_coeff(nb,ne)
        # Get random overlap matrix
        S  = self.get_random_overlap(nb)
        # Orthogonalise reference orbitals
        Cw = orthogonalise(Cw,S)
        Cx = orthogonalise(Cx,S)

        # Get initial overlap matrix
        Sold = np.einsum('pi,pq,qj',np.conj(Cw),S,Cx)

        # Perform the Lowdin-pairing
        Cw_new, Cx_new, Sxx = lowdin_pair(Cw,Cx,S)

        # Test if we actually have diagonal orbitals
        Snew = np.einsum('pi,pq,qj',np.conj(Cw_new),S,Cx_new)
        diag_test = Snew - np.diag(np.diag(Snew))
        self.assertTrue(np.linalg.norm(diag_test) < 1e-12, msg='Failed to biorthogonalise')

        # Test we have preserved the overlap
        self.assertTrue(abs(np.linalg.det(Snew) - np.linalg.det(Sold)) < 1e-12, msg='Failed to conserve overlap')


if __name__=='__main__':
    unittest.main()
