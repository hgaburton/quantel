#!/usr/bin/python3 

from functools import reduce
import numpy as np
from scipy.linalg import expm as scipy_expm


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

def stable_eigh(M,tol=1e-14):
    """Only diagonalise matrix M if off-diagonal elements are not small"""
    # Return nothing if matrix has size 0
    if(M.size == 0):
        return np.array([]), np.array([])
    if(np.max(np.abs(M - np.diag(np.diag(M)))) > tol):
        return np.linalg.eigh(M)
    else:
        return np.diag(M), np.eye(M.shape[0])
            
def random_rot(n, lmin, lmax):
    """
    Generate a random rotation matrix of size n x n.

    Parameters:
    - n (int): The size of the rotation matrix.
    - lmin (float): The minimum value for the random matrix elements.
    - lmax (float): The maximum value for the random matrix elements.

    Returns:
    - numpy.ndarray: The random rotation matrix.
    """
    X = lmin + np.random.rand(n,n) * (lmax - lmin)
    X = np.tril(X)  - np.tril(X).T
    return scipy_expm(X)

def delta_kron(i, j):
    """
    Returns the Kronecker delta function value for the given indices.

    Parameters:
    i (int): The first index.
    j (int): The second index.

    Returns:
    int: The value of the Kronecker delta function.
    """
    if i == j:
        return 1
    else:
        return 0
def delta_kron(i,j):
    if i==j: return 1
    else: return 0

def sym_orthogonalise(mat, metric, thresh=1e-10):
    """
    Orthogonalizes a matrix with respect to a symmetric metric.

    Parameters:
    - mat: The matrix to be orthogonalized.
    - metric: The symmetric metric used for orthogonalization.
    - thresh: The threshold value for eigenvalue truncation (default: 1e-10).

    Returns:
    - The orthogonalized matrix.
    """
    S = mat.T.dot(metric.dot(mat))
    eigval, eigvec = np.linalg.eigh(S)
    X = eigvec.dot(np.diag(np.power(eigval,-0.5)))
    return mat.dot(X)


def norm(v, metric=None):
    '''
    Get the norm of a vector for different metric tensor

    Parameters:
        v (ndarray)      : Vector input.
        metric (ndarray) : The metric tensor used for orthogonalisation.

    Returns:
        float : The norm of the vector
    '''
    if(metric is None):
        return np.sqrt(np.dot(v.conj().T,v))
    else:
        return np.sqrt(np.dot(v.conj().T,metric.dot(v)))


def inner_prod(a,b,metric=None):
    '''
    Get the inner product of two matrices with a given a metric tensor a.T @ S @ b

    Parameters:
        a (ndarray)      : Matrix bra states.
        b (ndarray)      : Matrix ket states.
        metric (ndarray) : The metric tensor used for orthogonalisation.

    Returns:
        ndarray : The result a.T @ S @ b
    '''
    if(metric is None):
        return np.dot(a.conj().T,b)
    else:
        return np.dot(a.conj().T,metric.dot(b))


def gram_schmidt(mat, metric=None):
    '''
    Perform Gram-Schmidt orthogonalisation for a set of vectors with respect to a metric S.
    
    Parameters: 
        mat (ndarray)    : Matrix with columns containing the vectors to be orthogonlised
        metric (ndarray) : Metric tensor used for the orthogonalisation  
                           Default None corresponds to an orthogonal space

    Returns: 
        ndarray          : Matrix containing the orthogonalised vectors
    '''
    # Copy of metric @ vector for use later
    Sv = np.copy(mat) if (metric is None) else metric @ mat

    # Orthogonalise each vector sequentially
    for i in range(0,mat.shape[1]):
        # Target vector
        vi  = mat[:,i] 
        # Projection space
        vj  = mat[:,:i]
        # Perform the projection
        vi -= vj @ (vj.conj().T @ Sv[:,i])
        # Renormalise
        vi /= norm(vi,metric)
    return mat


def orthogonalise(mat, metric=None, thresh=1e-10, fill=True):
    '''
    Orthogonalise the columns of mat with respect to the metric tensor.

    Parameters:
        mat (ndarray): The matrix whose columns need to be orthogonalised.
        metric (ndarray): The metric tensor used for orthogonalisation.
        thresh (float, optional): The threshold for the orthogonality test. Defaults to 1e-10.
        fill (bool, optional): Whether to fill the matrix with random values if the number of columns is less than the metric tensor. Defaults to True.

    Returns:
        ndarray: The orthogonalised matrix.
    '''
    # Pad the output matrix if requested
    (nr,nc) = mat.shape
    if(nc < nr and fill):
        new_mat = np.zeros((nr,nr)) if(metric is None) else np.zeros(metric.shape)
        new_mat[:,:nc] = mat.copy()
        new_mat[:,nc:] = np.random.rand(mat.shape[0],new_mat.shape[1]-nc)
        mat = new_mat

    # Check the orthogonality
    ortho = inner_prod(mat,mat,metric)
    for i in range(mat.shape[1]):
        ortho[i,i] -= 1.0
    ortho_test = np.max(np.abs(ortho))

    # If the orthogonality test fails, we need to orthogonalise
    if ortho_test > thresh:
        # Perform Gram-Schmidt twice for stability
        mat = gram_schmidt( gram_schmidt(mat,metric),metric )

        # Re-check the orthogonality
        ortho = inner_prod(mat,mat,metric)
        for i in range(mat.shape[1]):
            ortho[i,i] -= 1.0
        ortho_test = np.max(np.abs(ortho))

        if(ortho_test > thresh):
            print(f"WARNING: Gram-Schmidt orthogonalisation failed with max(abs(error)) = {ortho_test: 7.3e}")

    return mat

def matrix_print(M, title=None, ncols=6, offset=0):
    '''Print a matrix in a nice format with ncols columns at a time'''
    # Total number of columns
    nc = M.shape[1]
    if title is not None:
        print("\n -----------------------------------------------------------------------------------------------")
        print("  "+title)
        print(" -----------------------------------------------------------------------------------------------")
    # Loop over output blocks
    for i in range(np.ceil(nc/ncols).astype(int)):
        print("     ",end="")
        for j in range(ncols):
            if(i*ncols+j < nc):
                print(f"{i*ncols+j+1+offset:^14d} ",end="")
        print()
        for irow, row in enumerate(M[:,i*ncols:min(nc,(i+1)*ncols)]):
            print(f"{irow+1: 4d} ",end="")
            for val in row:
                print(f"{val:^14.8f} ",end="")
            print()
