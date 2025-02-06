#ifndef INTEGRAL_INTERFACE_H
#define INTEGRAL_INTERFACE_H

#include <vector>
#include <cassert>
#include <iostream>

class IntegralInterface {
    /// \brief IntegralInterface class
    /// \details This class provides an interface for numerical integration

protected:
    // Information about the basis
    size_t m_nbsf; //!< Number of basis functions
    size_t m_nmo; //!< Number of molecular orbitals

    /// Constant scalar potential
    double m_V;

    /// Overlap matrix
    std::vector<double> m_S; 
    /// Orthogonalisation matrix
    std::vector<double> m_X;

    /// One-electron integrals
    std::vector<double> m_oei_a;
    std::vector<double> m_oei_b;
    /// Dipole integrals
    std::vector<double> m_dipole;
    /// Store J and K versions of two-electron integrals
    std::vector<double> m_tei; /// [p,q,r,s] = (pq|rs)
    double thresh = 1e-12;


public:
    /** \brief Destructore for the interface 
     **/ 
    virtual ~IntegralInterface() { }

    /** \brief Default constructor 
     **/
    IntegralInterface(size_t nbsf) : m_nbsf(nbsf) { }

    /// Set the scalar potential
    virtual void set_scalar_potential(double value);

    /// Set the value of overlap integrals
    /// @param p integral index for bra 
    /// @param q integral index for ket
    /// @param value value of the integral
    virtual void set_ovlp(size_t p, size_t q, double value);

    /// Set the value of one-electron integrals
    /// @param p integral index for bra 
    /// @param q integral index for ket
    /// @param value value of the integral
    /// @param alpha spin of the integral
    virtual void set_oei(size_t p, size_t q, double value, bool alpha);

    /// Set an element of the two-electron integrals (pq|rs)
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    /// @param value value of the integral
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value);

    /// Build fock matrix from restricted density matrix in AO basis
    /// @param D density matrix
    /// @param F output fock matrix
    virtual void build_fock(std::vector<double> &dens, std::vector<double> &fock);

    /// Build the JK matrix from the density matrix in the AO basis
    /// @param D density matrix
    /// @param JK output JK matrix
    virtual void build_JK(std::vector<double> &dens, std::vector<double> &JK);

    /// Build J and K matrices from a list of density matrices
    /// @param vDJ Vector of density matrices for J build
    /// @param vDK Vector of density matrices for K build
    /// @param vJ Vector of output J matrices
    /// @param vK Vector of output K matrices
    /// @param nj number of density matrices for J build
    /// @param nk number of density matrices for K build
    virtual void build_multiple_JK(std::vector<double> &vDJ, std::vector<double> &vDK, 
                           std::vector<double> &vJ, std::vector<double> &vK, 
                           size_t nj, size_t nk);

    /// Build a J matrix
    /// @param D density matrix
    /// @param J output J matrix
    virtual void build_J(std::vector<double> &dens, std::vector<double> &J);

    /// Get the value of the scalar potential
    virtual double scalar_potential() { return m_V; }

    /// Get an element of the overlap matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    virtual double overlap(size_t p, size_t q);

    /// Get an element of the one-electron Hamiltonian matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    /// @param alpha spin of the integral
    virtual double oei(size_t p, size_t q, bool alpha);

    /// Get an element of the two-electron integrals (pq|rs)
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    virtual double tei(size_t p, size_t q, size_t r, size_t s);

    /// Perform AO to MO eri transformation
    /// @param C1 transformation matrix
    /// @param C2 transformation matrix
    /// @param C3 transformation matrix
    /// @param C4 transformation matrix
    /// @param eri outpur array of two-electron integrals in physicist's notation
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    virtual void tei_ao_to_mo(std::vector<double> &C1, std::vector<double> &C2, 
                  std::vector<double> &C3, std::vector<double> &C4, 
                  std::vector<double> &eri, bool alpha1, bool alpha2);

    /// Perform AO to MO transformation for one-electron integrals
    /// @param C1 transformation matrix
    /// @param C2 transformation matrix
    /// @param oei_mo output array of one-electron integrals
    /// @param alpha spin of the integrals
    virtual void oei_ao_to_mo(std::vector<double> &C1, std::vector<double> &C2, 
                      std::vector<double> &oei_mo, bool alpha);

    /// @brief Perform AO to MO transformation for dipole integrals
    /// @param C1 transformation matrix
    /// @param C2 transformation matrix 
    /// @param dipole_mo output array of dipole integrals
    /// @param alpha spin of the integrals
    virtual void dipole_ao_to_mo(std::vector<double> &C1, std::vector<double> &C2, 
                      std::vector<double> &dipole_mo, bool alpha);

    // @brief Compute the orthogonalization matrix
    virtual void compute_orthogonalization_matrix();

    /// Get a pointer to the overlap matrix
    double *overlap_matrix() { return m_S.data(); }

    /// Get a pointer to the orthogonalisation matrix
    double *orthogonalization_matrix() { return m_X.data(); }

    /// Get a pointer to the one-electron Hamiltonian matrix
    /// @param alpha spin of the integrals
    double *oei_matrix(bool alpha) { return alpha ? m_oei_a.data() : m_oei_b.data(); }

    /// Get a pointer to the two-electron integral array
    double *tei_array() { return m_tei.data(); }

    /// Get a pointer to the dipole integrals
    double *dipole_integrals() { return m_dipole.data(); }

    /// Get the number of basis functions
    size_t nbsf() const { return m_nbsf; }

    /// Get the number of linearly indepdent molecular orbitals
    size_t nmo() const { return m_nmo; }

    /// Get index-for one-electron quantity
    size_t oei_index(size_t p, size_t q) 
    { 
        assert(p<m_nbsf);
        assert(q<m_nbsf);
        return p * m_nbsf + q; 
    }
    /// Get index-for two-electron quantity
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) 
    {
        assert(p<m_nbsf);
        assert(q<m_nbsf);
        assert(r<m_nbsf);
        assert(s<m_nbsf);
        return p * m_nbsf * m_nbsf * m_nbsf +q * m_nbsf * m_nbsf + r * m_nbsf + s;
    }
};

#endif // INTEGRAL_INTERFACE_H