#!/usr/bin/python3 

from functools import reduce
import numpy as np
from scipy.linalg import expm as scipy_expm

def stable_eigh(M,tol=1e-14):
    """Only diagonalise matrix M if off-diagonal elements are not small"""
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

def orthogonalise(mat, metric, thresh=1e-10, fill=True):
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
    nc = mat.shape[1]

    if nc < metric.shape[1] and fill:
        new_mat = 0.0 * metric
        new_mat[:,:nc] = mat.copy()
        new_mat[:,nc:] = np.random.rand(mat.shape[0],new_mat.shape[1]-nc)
        mat = new_mat

    ortho = reduce(np.dot, (mat.conj().T, metric, mat))
    ortho_test = np.linalg.norm(ortho - np.identity(ortho.shape[0])) / np.sqrt(ortho.size)
    if ortho_test > thresh:
        # First Gram-Schmidt
        proj = np.zeros(metric.shape)
        for i in range(0,mat.shape[1]):
            mat[:,i] -= proj.dot(metric.dot(mat[:,i]))
            norm = reduce(np.dot, (mat[:,i].T, metric, mat[:,i]))
            mat[:,i] /= np.sqrt(norm)
            proj     += np.outer(mat[:,i], mat[:,i])

        # Second Gram-Schmidt
        proj = np.zeros(metric.shape)
        for i in range(0,mat.shape[1]):
            mat[:,i] -= proj.dot(metric.dot(mat[:,i]))
            norm = reduce(np.dot, (mat[:,i].T, metric, mat[:,i]))
            mat[:,i] /= np.sqrt(norm)
            proj     += np.outer(mat[:,i], mat[:,i])

        # Double-check the normalisation
        for i in range(mat.shape[1]):
            norm = reduce(np.dot, (mat[:,i].T, metric, mat[:,i]))
            mat[:,i] /= np.sqrt(reduce(np.dot, (mat[:,i], metric, mat[:,i])))
      
    return mat
