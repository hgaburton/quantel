#!/usr/bin/python3 

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
        V = np.asmatrix(VT).H

        # Transform the orbitals
        Cw_new = np.einsum('pi,ij->pj',Cw,U)
        Cx_new = np.einsum('pi,ij->pj',Cw,V)

        # Correct for any new phase
        Cw_new[:,0] *= np.linalg.det(np.asmatrix(U).H)
        Cx_new[:,0] *= np.linalg.det(np.asmatrix(V).H)

        # Compute new overlap matrix
        Swx = np.einsum('pi,pq,qj->ij',np.conj(Cw_new),S,Cx_new)

    # Get diagonal of overlap matrix
    Sxx = np.diag(Swx)

    # Return the result
    return Cw_new, Cx_new, Sxx


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
