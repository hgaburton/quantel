#!/usr/bin/python3 

from functools import reduce
import numpy as np

def delta_kron(i,j):
    if i==j: return 1
    else: return 0

def orthogonalise(mat, metric, thresh=1e-10, fill=True):
    '''Orthogonalise the columns of mat with respect to the metric tensor'''
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
