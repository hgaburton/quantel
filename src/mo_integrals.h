#ifndef MO_INTEGRALS_H
#define MO_INTEGRALS_H

#include <vector>
#include "two_array.h"
#include "four_array.h"

class MOintegrals {
private:
    /// Number of correlated orbitals
    size_t m_nmo;
    /// Scalar potential
    double m_V;
    /// Tolerance for integral screening
    double m_tol;
    /// NOTE: Currently supports only restricted integrals
    /// One-electron MO integrals
    TwoArray m_oei;
    /// Two-electron MO integrals <pq|rs> without antisymmetrization
    FourArray m_tei;

public:
    /// \brief Default destructor
    virtual ~MOintegrals() { }

    /// \brief Constructor from arrays
    /// \param scalar_potential Constant scalar potential
    /// \param oei One-electron integral object as TwoArray
    /// \param tei Two-electron integral object as FourArray
    /// \param nmo Number of molecular orbitals
    /// \param tolerance Integral screening tolerance [default: 1e-14]
    MOintegrals(
        double scalar_potential,
        TwoArray oei, FourArray tei, size_t nmo, double tolerance = 1e-14) :
        m_oei(oei), m_tei(tei),m_V(scalar_potential), m_tol(tolerance), m_nmo(nmo)
    { 
        // Check dimensions of one-electron integrals
        if(oei.dim() != std::make_tuple(m_nmo, m_nmo))
            throw std::invalid_argument("Dimensions of oei do not match number of MOs.");

        // Check dimensions of two-electron integrals
        if(tei.dim() != std::make_tuple(m_nmo, m_nmo, m_nmo, m_nmo))
            throw std::invalid_argument("Dimensions of tei do not match number of MOs.");
    }

    /// \brief Constructor from vectors
    /// \param scalar_potential Constant scalar potential
    /// \param oei_data One-electron integral data as std::vector<double>
    /// \param tei_data Two-electron integral data as std::vector<double>
    /// \param nmo Number of molecular orbitals
    /// \param tolerance Integral screening tolerance [default: 1e-14]
    MOintegrals(
        double scalar_potential,
        std::vector<double> oei_data,std::vector<double> tei_data,
        size_t nmo,double tolerance = 1e-14) :
        m_oei(oei_data,nmo,nmo),m_tei(tei_data,nmo,nmo,nmo,nmo),
        m_V(scalar_potential),m_tol(tolerance),m_nmo(nmo)
    { 
        // Check dimensions of one-electron integrals
        if(oei_data.size() != nmo * nmo)
            throw std::invalid_argument("Size of oei_data does not match number of MOs.");

        // Check dimensions of two-electron integrals
        if(tei_data.size() != nmo * nmo * nmo * nmo)
            throw std::invalid_argument("Size of tei_data does not match number of MOs.");
    }

    /// \brief Get the value of the scalar potential
    double scalar_potential() const { return m_V; }

    /// \brief Get an element of the one-electron Hamiltonian matrix
    /// @param p integral index for bra
    /// @param q integral index for ket
    double oei(size_t p, size_t q) 
    {
        return m_oei(p,q);
    }

    /// \brief Get an element of the two-electron integrals <pq||rs>
    /// @param p integral index
    /// @param q integral index
    /// @param r integral index 
    /// @param s integral index
    double tei(size_t p, size_t q, size_t r, size_t s)
    {
        return m_tei(p,q,r,s);
    }

    /// \brief Get a pointer to the one-electron Hamiltonian matrix
    TwoArray *oei_matrix() { return &m_oei; }

    /// Get a point to the two-electron integral array
    FourArray *tei_array() { return &m_tei; }

    /// \brief Get integral screening threshold
    double tol() const { return m_tol; }

    /// Number of MOs
    size_t nmo() const { return m_nmo; }

};

#endif // MO_INTEGRALS_H