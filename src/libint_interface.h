#ifndef LIBINT_INTERFACE_H
#define LIBINT_INTERFACE_H

#include <libint2.hpp>
#include "molecule.h"

class LibintInterface {
private:
    const libint2::BasisSet m_basis; //!< The Libint2 basis set
    const Molecule &m_mol;  

    // Information about the basis
    size_t m_nbsf; //!< Number of basis functions

    /// Constant scalar potential
    double m_V;
    /// Overlap matrix
    std::vector<double> m_S; 
    /// One-electron integrals
    std::vector<double> m_oei_a;
    std::vector<double> m_oei_b;
    /// Two-electron integrals
    std::vector<double> m_tei_aa;
    std::vector<double> m_tei_bb;
    std::vector<double> m_tei_ab;

public:

    /** \brief Destructor for the interface
     **/
    virtual ~LibintInterface() { }

    /** \brief Constructor for the interface
     **/
    LibintInterface(const std::string basis_str, Molecule &mol) :
        m_mol(mol), m_basis(basis_str, mol.atoms)
    { 
        initialize();
    }

    /// Set the scalar potential
    void set_scalar_potential(double value);

    /// Set the value of overlap integrals
    /// @param p integral index for bra 
    /// @param q integral index for ket
    /// @param value value of the integral
    void set_ovlp(size_t p, size_t q, double value);

    /// Set the value of one-electron integrals
    /// @param p integral index for bra 
    /// @param q integral index for ket
    /// @param value value of the integral
    /// @param alpha spin of the integral
    void set_oei(size_t p, size_t q, double value, bool alpha);

    /// Set the value of two-electron integrals (antisymmetrised physicist notation)
    /// @param p integral index for bra 
    /// @param q integral index for ket
    /// @param value value of the integral
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1, bool alpha2);

    /// Get the number of basis functions
    size_t nbsf() { return m_nbsf; }
    
    /// Get the value of the scalar potential
    double scalar_potential() { return m_V; }

    /// Get an element of the overlap matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    double overlap(size_t p, size_t q);

    /// Get an element of the one-electron Hamiltonian matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    /// @param alpha spin of the integral
    double oei(size_t p, size_t q, bool alpha);

    /// Get an element of the two-electron integrals <pq||rs>
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    double tei(size_t p, size_t q, size_t r, size_t s, bool alpha1, bool alpha2);

    /// Get a pointer to the overlap matrix
    double *overlap_matrix() { return m_S.data(); }

    /// Get a pointer to the one-electron Hamiltonian matrix
    /// @param alpha spin of the integrals
    double *oei_matrix(bool alpha) { return alpha ? m_oei_a.data() : m_oei_b.data(); }


    /** \brief Compute two-electron integrals **/
    //void two_electron_matrix(four_array<double> &II);

    /** Initialise all relevant variabes **/
    void initialize();


private:
    /// Compute the nuclear repulsion energy
    void compute_nuclear_potential();
    /// Compute overlap integrals
    void compute_overlap();
    /// Compute the one-electron Hamiltonian integrals
    void compute_one_electron_matrix();
    /// Compute the two-electron integrals
    void compute_two_electron_integrals();

    /// Get index-for one-electron quantity
    size_t oei_index(size_t p, size_t q) { 
        return p * m_nbsf + q; 
    }
    /// Get index-for two-electron quantity
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) { 
        return p * m_nbsf * m_nbsf * m_nbsf +q * m_nbsf * m_nbsf + r * m_nbsf + s;
    }
};

#endif // LIBINT_INTERFACE_H
