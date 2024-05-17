#include "linalg.h"
#include <cassert>

size_t orthogonalisation_matrix(
    size_t dim, std::vector<double> &M, double thresh, std::vector<double> &X)
{
    // Check input
    assert(M.size() == dim * dim);

    // Convert to Armadillo matrices
    arma::mat Mmat(M.data(), dim, dim, false, true);

    // Need an inplace transpose because Armadillo uses column-major order
    arma::inplace_trans(Mmat);
    
    // Diagonalise
    arma::mat eigvec;
    arma::vec eigval;
    if(!arma::eig_sym(eigval, eigvec, Mmat, "std"))
    {
        throw std::runtime_error("orthogonalisation_matrix: Unable to diagonalise M matrix");
    }
    
    // Remove null space
    size_t null_dim=0;
    while(eigval(null_dim) <= thresh) null_dim++;
    arma::vec eprime(dim - null_dim);
    for(size_t i=null_dim; i < dim; i++)
        eprime(i-null_dim) = 1.0 / std::sqrt(eigval(i));

    // Construct orthogonalisation matrix
    X.resize(dim * (dim - null_dim), 0.0);
    arma::mat Xmat = arma::mat(X.data(), dim, dim - null_dim, false, true);
    Xmat = eigvec.cols(null_dim, dim-1) * arma::diagmat(eprime);

    // Need an inplace transpose because Armadillo uses column-major order
    arma::inplace_trans(Xmat);
    arma::inplace_trans(Mmat);

    return dim - null_dim;
}

void gen_eig_sym(
    const size_t dim, 
    std::vector<double> &M, std::vector<double> &S, std::vector<double> &X, 
    std::vector<double> &eigval, std::vector<double> &eigvec, 
    double thresh)
{
    // Check the input
    assert(M.size() == dim * dim);
    assert(S.size() == dim * dim);
    
    // Solve the generalised eigenvalue problem
    size_t n_span = orthogonalisation_matrix(dim, S, thresh, X);

    // Convert to Armadillo matrices
    arma::mat Mmat = arma::mat(M.data(), dim, dim, false, true);
    arma::mat Xmat = arma::mat(X.data(), dim, n_span, false, true);

    // Need an inplace transpose because Armadillo uses column-major order
    arma::inplace_trans(Mmat);
    arma::inplace_trans(Xmat);

    // Resize output
    eigval.resize(dim);
    eigvec.resize(dim * n_span);
    arma::vec vec_eigval = arma::vec(eigval.data(), dim, false, true);
    arma::mat mat_eigvec = arma::mat(eigvec.data(), dim, n_span, false, true);

    // Transform to orthogonal space
    arma::mat ortho_M = Xmat.t() * Mmat * Xmat;
    arma::eig_sym(vec_eigval, mat_eigvec, ortho_M, "dc");

    // Transform back to original space
    mat_eigvec = Xmat * mat_eigvec;

    // Sort eigenvalues small to high
    mat_eigvec = mat_eigvec.cols(arma::stable_sort_index(vec_eigval));
    vec_eigval = arma::sort(vec_eigval);

    // Need an inplace transpose because Armadillo uses column-major order
    arma::inplace_trans(Mmat);
    arma::inplace_trans(Xmat);
    arma::inplace_trans(mat_eigvec);
}