#ifndef MO_INTEGRALS_H
#define MO_INTEGRALS_H

#include <vector>
#include "libint_interface.h"

/// TODO: Add frozen core option

class MOintegrals {
public:
    /// Default destructor
    virtual ~MOintegrals() { }

    /// Constructor from orbital coefficients and integrals
    MOintegrals(std::vector<double> &Ca, std::vector<double> &Cb, LibintInterface &ints) :
        m_Ca(Ca), m_Cb(Cb), m_ints(ints), m_nbsf(ints.nbsf()), m_nmo(ints.nmo())
    { 
        // Check dimensions
        assert(m_Ca.size() == m_nbsf * m_nmo);
        assert(m_Cb.size() == m_nbsf * m_nmo);
        // Initialize integral values
        initialize();
    }

    /// Get the value of the scalar potential
    double scalar_potential() { return m_V; }

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

    /// Get a pointer to the one-electron Hamiltonian matrix
    /// @param alpha spin of the integrals
    double *oei_matrix(bool alpha) { return alpha ? m_oei_a.data() : m_oei_b.data(); }

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

    /// Get the number of basis functions
    size_t nbsf() const { return m_nbsf; }
    /// Get the number of linearly indepdent molecular orbitals
    size_t nmo() const { return m_nmo; }

private:
    /// Orbital coefficients
    std::vector<double> &m_Ca;
    std::vector<double> &m_Cb;

    /// Number of basis functions
    size_t m_nbsf;
    /// Number of molecular orbitals
    size_t m_nmo;

    /// Libint interface
    LibintInterface &m_ints;

    /// Scalar potential 
    double m_V;
    /// One-electron MO integrals
    std::vector<double> m_oei_a;
    std::vector<double> m_oei_b;
    /// Two-electron MO integrals
    std::vector<double> m_tei_aa;
    std::vector<double> m_tei_bb;
    std::vector<double> m_tei_ab;

    /// Initialise
    void initialize();

    /// Compute scalar potential
    void compute_scalar_potential();
    /// Compute one-electron integrals
    void compute_oei(bool alpha);
    /// Compute two-electron integrals
    void compute_tei(bool alpha1, bool alpha2);

    /// Get index-for one-electron quantity
    size_t oei_index(size_t p, size_t q) 
    { 
        assert(p<m_nmo);
        assert(q<m_nmo);
        return p * m_nmo + q; 
    }
    /// Get index-for two-electron quantity
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) 
    {
        assert(p<m_nmo);
        assert(q<m_nmo);
        assert(r<m_nmo);
        assert(s<m_nmo);
        return p * m_nmo * m_nmo * m_nmo + q * m_nmo * m_nmo + r * m_nmo + s;
    }
};

#endif // MO_INTEGRALS_H