#ifndef LIBINT_INTERFACE_H
#define LIBINT_INTERFACE_H

#include <libint2.hpp>
#include "molecule.h"

class LibintInterface {
    /// \brief LibintInterface class
    /// \details This class provides an interface to the Libint2 library for computing molecular integrals
 
private:
    const libint2::BasisSet m_basis; //!< The Libint2 basis set
    const Molecule &m_mol;

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

    /// Return pointer to m_mol
    const Molecule &molecule() { return m_mol; }

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

    /// Set an element of the two-electron integrals (pq|rs)
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    /// @param value value of the integral
    void set_tei(size_t p, size_t q, size_t r, size_t s, double value);

    /// Build fock matrix from restricted density matrix in AO basis
    /// @param D density matrix
    /// @param F output fock matrix
    void build_fock(std::vector<double> &dens, std::vector<double> &fock);

    /// Build the JK matrix from the density matrix in the AO basis
    /// @param D density matrix
    /// @param JK output JK matrix
    void build_JK(std::vector<double> &dens, std::vector<double> &JK);

    /// Build J and K matrices from a list of density matrices
    /// @param vDJ Vector of density matrices for J build
    /// @param vDK Vector of density matrices for K build
    /// @param vJ Vector of output J matrices
    /// @param vK Vector of output K matrices
    /// @param nj number of density matrices for J build
    /// @param nk number of density matrices for K build
    void build_multiple_JK(std::vector<double> &vDJ, std::vector<double> &vDK, 
                           std::vector<double> &vJ, std::vector<double> &vK, 
                           size_t nj, size_t nk);

    /// Build a J matrix
    /// @param D density matrix
    /// @param J output J matrix
    void build_J(std::vector<double> &dens, std::vector<double> &J);

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

    /// Get an element of the two-electron integrals (pq|rs)
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    double tei(size_t p, size_t q, size_t r, size_t s);

    /// Perform AO to MO eri transformation
    /// @param C1 transformation matrix
    /// @param C2 transformation matrix
    /// @param C3 transformation matrix
    /// @param C4 transformation matrix
    /// @param eri outpur array of two-electron integrals in physicist's notation
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    void tei_ao_to_mo(std::vector<double> &C1, std::vector<double> &C2, 
                  std::vector<double> &C3, std::vector<double> &C4, 
                  std::vector<double> &eri, bool alpha1, bool alpha2);

    /// Perform AO to MO transformation for one-electron integrals
    /// @param C1 transformation matrix
    /// @param C2 transformation matrix
    /// @param oei_mo output array of one-electron integrals
    /// @param alpha spin of the integrals
    void oei_ao_to_mo(std::vector<double> &C1, std::vector<double> &C2, 
                      std::vector<double> &oei_mo, bool alpha);

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

    /// Initialise all relevant variables
    void initialize();

    /// Plot orbitals
    void molden_orbs(std::vector<double> &C, 
                     std::vector<double> &occ, 
                     std::vector<double> &evals);

private:
    /// Compute the nuclear repulsion energy
    void compute_nuclear_potential();
    /// Compute overlap integrals
    void compute_overlap();
    /// Compute orthogonalization matrix
    void compute_orthogonalization_matrix();
    /// Compute the one-electron Hamiltonian integrals
    void compute_one_electron_matrix();
    /// Compute the two-electron integrals
    void compute_two_electron_integrals();
    /// Compute the dipole integrals
    void compute_dipole_integrals();

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

#endif // LIBINT_INTERFACE_H
