#ifndef MO_INTEGRALS_H
#define MO_INTEGRALS_H

#include <vector>
#include "libint_interface.h"

/// TODO: Add frozen core option

class MOintegrals {
public:
    /// \brief Default destructor
    virtual ~MOintegrals() { }

    /// \brief Constructor from integrals
    MOintegrals(LibintInterface &ints) :
       m_ints(ints), m_nbsf(ints.nbsf()), m_nmo(ints.nmo())
    { }

    /// \brief Compute integrals from orbital coefficients
    /// \param Ca Coefficients for alpha orbitals
    /// \param Cb Coefficients for beta orbitals
    /// \param nin Number of inactive orbitals
    /// \param nvr Number of virtual orbitals
    void update_orbitals(std::vector<double> C, size_t ninactive, size_t nvirtual);


    /// \brief Get the value of the scalar potential
    double scalar_potential() const { return m_V; }

    /// \brief Get an element of the one-electron Hamiltonian matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    /// @param alpha spin of the integral
    double oei(size_t p, size_t q, bool alpha);

    /// \brief Get an element of the two-electron integrals <pq||rs>
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    double tei(size_t p, size_t q, size_t r, size_t s, bool alpha1, bool alpha2);

    /// \brief Get a pointer to the one-electron Hamiltonian matrix
    /// @param alpha spin of the integrals
    double *oei_matrix(bool alpha) { return alpha ? m_oei_a.data() : m_oei_b.data(); }

    /// \breif Get a point to the dipole matrix
    /// @param alpha spin of the integrals
    double *dipole_matrix(bool alpha) { return m_dip.data(); }

    /// Get a point to the two-electron integral array
    /// @param alpha1 spin of electron 1
    /// @param alpha2 spin of electron 2
    double *tei_array(bool alpha1, bool alpha2) { 
        if(alpha1 == true and alpha2 == true)
            return m_tei_aa.data();
        if(alpha1 == true and alpha2 == false)
            return m_tei_ab.data();
        if(alpha1 == false and alpha2 == false)    
            return m_tei_bb.data();
        return nullptr;
    }

    /// \brief Get the number of basis functions
    size_t nbsf() const { return m_nbsf; }
    /// \brief Get the number of linearly indepdent molecular orbitals
    size_t nmo() const { return m_nmo; }
    /// \brief Get the number of correlated orbitals
    size_t nact() const { return m_nact; }
    /// \brief Get integral screening threshold
    double tol() const { return m_tol; }

private:
    /// Orbital coefficients
    std::vector<double> m_C;
    std::vector<double> m_Cact;
    /// Core density in AO basis
    std::vector<double> m_Pcore;
    /// Inactive one-electron potential
    std::vector<double> m_Vc_oei; 

    /// Number of basis functions
    size_t m_nbsf = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;
    /// Number of correlated orbitals
    size_t m_nact = 0;
    // Number of inactive orbitals
    size_t m_ncore = 0;

    /// Intergral screening threshold
    double m_tol = 1e-14;

    /// Libint interface
    LibintInterface &m_ints;

    /// Scalar potential 
    double m_V;
    /// Scalar core potential
    double m_Vc = 0;
    /// Scalar dipole contribution
    std::vector<double> m_dipC;

    /// One-electron MO integrals
    std::vector<double> m_oei_a;
    std::vector<double> m_oei_b;
    /// Two-electron MO integrals
    std::vector<double> m_tei_aa;
    std::vector<double> m_tei_bb;
    std::vector<double> m_tei_ab;

    /// Dipole MO integrals
    std::vector<double> m_dip;

    /// \brief Compute core density
    void compute_core_density();
    /// \brief Compute the effective core potential
    void compute_core_potential();
    /// \brief Compute scalar potential
    void compute_scalar_potential();
    /// \brief Compute one-electron integrals
    void compute_oei(bool alpha);
    /// \brief Compute two-electron integrals
    void compute_tei(bool alpha1, bool alpha2);
    /// \brief Compute core dipole contribution
    void compute_core_dipole();
    /// \brief Compute dipole integrals
    void compute_dipole(bool alpha); 

    /// \brief Get index-for one-electron quantity
    size_t oei_index(size_t p, size_t q) 
    { 
        assert(p<m_nact);
        assert(q<m_nact);
        return p * m_nact + q; 
    }
    /// \brief Get index-for two-electron quantity
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) 
    {
        assert(p<m_nact);
        assert(q<m_nact);
        assert(r<m_nact);
        assert(s<m_nact);
        return p * m_nact * m_nact * m_nact + q * m_nact * m_nact + r * m_nact + s;
    }
};

#endif // MO_INTEGRALS_H
