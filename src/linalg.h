#ifndef LINALG_H_
#define LINALG_H_

#include <cstddef>
#include <armadillo>

/// \brief Compute the orthogonalisation transformation for a matrix
/// \param dim Dimensions of input matrix
/// \param M Input matrix
/// \param thresh Threshold for zero eigenvalues
/// \param[out] X Orthogonalisation matrix
size_t orthogonalisation_matrix(
    size_t dim, std::vector<double> &M, double thresh, std::vector<double> &X);

/// \brief Solve the generalised eigenvalue problem
/// \param dim Dimensions of the eigenvalue target matrices
/// \param M Matrix to be diagonalised
/// \param S Corresponding overlap matrix of generalised eigenvalue problem
/// \param X Orthogonalisation transformation matrix to be identified
/// \param eigval Vector of eigenvalues
/// \param eigvec Matrix with eigenvectors as columns
/// \param thresh Threshold for null space in overlap matrix
void gen_eig_sym(
    const size_t dim, 
    std::vector<double> &M, std::vector<double> &S, std::vector<double> &X, 
    std::vector<double> &eigval, std::vector<double> &eigvec, double thresh=1e-8);

#endif // LINALG_H_