#ifndef FCIDUMP_INTERFACE_H
#define FCIDUMP_INTERFACE_H

#include <string>
#include <vector>
#include <stdexcept>
#include "mo_integrals.h"
#include "linalg.h"

/// \brief FCIDumpInterface class
/// \details Provides an integral interface backed by a pre-computed FCIDUMP file.
///          The orbitals in the FCIDUMP are assumed to be orthonormal, so the
///          overlap and orthogonalisation matrices are both the identity. The
///          stored AO-labelled arrays (m_oei_a, m_tei, …) actually hold MO
///          integrals; coefficient matrices passed to the transformation routines
///          are therefore rotations within the orbital space.
///          Assume that the FCIDUMP is in restricted orbitals
class FCIDumpInterface {

protected:
    size_t m_nbsf;   ///< Number of orbitals (= NORB from header)
    size_t m_nmo;    ///< Same as m_nbsf (all orbitals are linearly independent)
    size_t m_nelec;  ///< Total number of electrons (NELEC from header)
    size_t m_ms2;    ///< 2*Sz (MS2 from header)

    double m_V;                  ///< Scalar potential / nuclear repulsion energy
    std::vector<double> m_S;     ///< Overlap matrix (identity)
    std::vector<double> m_X;     ///< Orthogonalisation matrix (identity)
    std::vector<double> m_oei_a; ///< Alpha one-electron integrals h(p,q)
    std::vector<double> m_oei_b; ///< Beta  one-electron integrals h(p,q)
    std::vector<double> m_tei;   ///< Two-electron integrals (pq|rs), chemist's notation

    double thresh = -1; ///< Screening threshold used in JK builds and transformations

public:
    virtual ~FCIDumpInterface() { }

    /// \brief Construct interface by reading a FCIDUMP file
    /// \param filename Path to the FCIDUMP file
    FCIDumpInterface(const std::string &filename) { initialize(filename); }

    /// Number of basis functions (= number of orbitals in the FCIDUMP)
    size_t nbsf()  const { return m_nbsf;  }
    /// Number of linearly independent MOs (= nbsf for an orthogonal FCIDUMP)
    size_t nmo()   const { return m_nmo;   }
    /// Total number of electrons
    size_t nelec() const { return m_nelec; }
    /// Number of alpha electrons
    size_t nalfa() const { return (m_nelec + m_ms2) / 2; }
    /// Number of beta electrons
    size_t nbeta() const { return (m_nelec - m_ms2) / 2; }

    /// Scalar potential (nuclear repulsion + frozen-core offset from FCIDUMP)
    double scalar_potential() { return m_V; }

    /// Pointer to the overlap matrix (identity for an orthogonal FCIDUMP)
    double *overlap_matrix()           { return m_S.data();    }
    /// Pointer to the orthogonalisation matrix (identity for an orthogonal FCIDUMP)
    double *orthogonalization_matrix() { return m_X.data();    }
    /// Pointer to the one-electron Hamiltonian matrix (alpha or beta)
    /// \param alpha True for alpha integrals, false for beta integrals
    double *oei_matrix(bool alpha)     { return alpha ? m_oei_a.data() : m_oei_b.data(); }
    /// Pointer to the two-electron integral array (pq|rs) in chemist's notation
    double *tei_array()                { return m_tei.data();  }

    /// Dipole integrals are not available from a FCIDUMP file
    double *dipole_integrals()
    {
        throw std::runtime_error("FCIDumpInterface: dipole integrals are not available from a FCIDUMP file");
    }

    /// Build restricted Fock matrix F = h + (2J-K) from a density matrix
    /// \param dens Density matrix in the same basis as the integrals
    /// \param fock Output Fock matrix
    void build_fock(std::vector<double> &dens, std::vector<double> &fock);

    /// Build (2J-K) from a single density matrix
    /// \param dens Density matrix in the same basis as the integrals
    /// \param JK Output matrix to hold (2J-K)
    virtual void build_JK(std::vector<double> &dens, std::vector<double> &JK);

    /// Build J and K matrices for separate sets of density matrices
    /// \param vDJ Vector of density matrices for J builds (size = nj * nbsf * nbsf)
    /// \param vDK Vector of density matrices for K builds (size = nk * nbsf * nbsf)
    /// \param vJ  Output vector of J matrices (size = nj * nbsf * nbsf)
    /// \param vK  Output vector of K matrices (size = nk * nbsf * nbsf)
    /// \param nj  Number of densities in vDJ
    /// \param nk  Number of densities in vDK
    virtual void build_multiple_JK(
        std::vector<double> &vDJ, std::vector<double> &vDK,
        std::vector<double> &vJ,  std::vector<double> &vK,
        size_t nj, size_t nk);

    /// Transform one-electron integrals from the orbital basis to a new MO basis
    /// \param C1  Orbital coefficients for the bra state
    /// \param C2  Orbital coefficients for the ket state
    /// \param oei_mo Output matrix for one-electron integrals
    /// \param alpha Spin of oei integrals to use. [true=alpha|false=beta]
    void oei_ao_to_mo(
        std::vector<double> &C1, std::vector<double> &C2,
        std::vector<double> &oei_mo, bool alpha);

    /// Transform two-electron integrals from the orbital basis to a new MO basis.
    /// Output eri is in physicist's notation <pq||rs> with optional antisymmetrisation.
    /// \param C1  Orbital coefficients for electron 1 bra state
    /// \param C2  Orbital coefficients for electron 2 bra state
    /// \param C3  Orbital coefficients for electron 1 ket state
    /// \param C4  Orbital coefficients for electron 2 ket state
    /// \param eri Output array for two-electron integrals in physicist's notation
    /// \param alpha1 Spin of electron 1 integrals to use. [true=alpha|false=beta]
    /// \param alpha2 Spin of electron 2 integrals to use. [true=alpha|false=beta]
    virtual void tei_ao_to_mo(
        std::vector<double> &C1, std::vector<double> &C2,
        std::vector<double> &C3, std::vector<double> &C4,
        std::vector<double> &eri, bool alpha1, bool alpha2);

    /// Build an MOintegrals object for the (ncore, nactive) orbital partition of C
    /// \param C       Orbital coefficients
    /// \param ncore   Number of core orbitals
    /// \param nactive Number of active orbitals
    MOintegrals mo_integrals(
        std::vector<double> &C, size_t ncore = 0, size_t nactive = 0);

    /// Parse the FCIDUMP file and initialise all internal arrays
    /// \param filename Path to the FCIDUMP file
    void initialize(const std::string &filename);

private:
    /// Parse the FCIDUMP file
    void parse_fcidump(const std::string &filename);

    /// Flat index for m_tei in chemist's notation: (pq|rs) = m_tei[tei_index(p,q,r,s)]
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) const
    {
        return p * m_nbsf * m_nbsf * m_nbsf
             + q * m_nbsf * m_nbsf
             + r * m_nbsf
             + s;
    }
};

#endif // FCIDUMP_INTERFACE_H
