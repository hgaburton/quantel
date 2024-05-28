#ifndef CI_SPACE_H
#define CI_SPACE_H

#include <cstddef>
#include <map>
#include <vector>
#include <tuple>
#include <iostream>
#include <cassert>
#include "determinant.h"
#include "excitation.h"
#include "mo_integrals.h"
#include "fmt/core.h"

typedef std::map<Eph, std::vector<std::tuple<size_t,size_t,int> > > mem_map_1;
typedef std::map<Epphh, std::vector<std::tuple<size_t,size_t,int> > > mem_map_2;

/// \brief Container for information about a CI space
class CIspace 
{
public:
    /// Destructor
    virtual ~CIspace() { }

    /// Constructor
    CIspace(MOintegrals &mo_ints, size_t norb, size_t nalfa, size_t nbeta, std::string citype="FCI") :
        m_ints(mo_ints), m_nmo(norb), m_nalfa(nalfa), m_nbeta(nbeta)
    { 
        if(m_nmo == 0)
            throw std::runtime_error("CIspace::CIspace: No active orbitals");
        if(m_nalfa > m_nmo)
            throw std::runtime_error("CIspace::CIspace: Invalid number of alpha electrons");
        if(m_nbeta > m_nmo)
            throw std::runtime_error("CIspace::CIspace: Invalid number of beta electrons");
        initialize(citype);
    }

    /// Print the determinant list
    virtual void print() const;
    /// Print a CI vector
    virtual void print_vector(const std::vector<double> &ci_vec, double tol) const;

    /// Get the number of determinants
    size_t ndet() const { return m_ndet; }
    size_t ndeta() const { return m_ndeta; }
    size_t ndetb() const { return m_ndetb; }
    /// Get the number of electrons
    size_t nalfa() const { return m_nalfa; }
    size_t nbeta() const { return m_nbeta; }
    /// Get the number of molecular orbitals
    size_t nmo() const { return m_nmo; }

    /// @brief Compute the sigma vector
    /// @param ci_vec Input CI vector
    /// @param sigma Output sigma vector
    void H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma);

    /// @brief Compute the Hamiltonian matrix
    /// @param Hmat
    void build_Hmat(std::vector<double> &Hmat);

    /// @brief Compute 1RDM
    /// @param bra CI vector for bra
    /// @param ket CI vector for ket
    /// @param rdm1 Output 1RDM
    /// @param alpha Spin of the RDM
    void build_rdm1(
        const std::vector<double> &bra, 
        const std::vector<double> &ket, 
        std::vector<double> &rdm1, 
        bool alpha);

    /// @brief Compute 2RDM
    /// @param bra CI vector for bra
    /// @param ket CI vector for ket
    /// @param rdm2 Output 2RDM
    /// @param alpha1 Spin of electron 1
    /// @param alpha2 Spin of electron 2
    void build_rdm2(
        const std::vector<double> &bra, const std::vector<double> &ket, 
        std::vector<double> &rdm2, bool alpha1, bool alpha2);

private:
    /// MO integrals
    MOintegrals &m_ints;

    /// Number of determinants
    size_t m_ndet = 0;
    size_t m_ndeta = 0;
    size_t m_ndetb = 0;
    /// Number of electrons
    size_t m_nalfa = 0;
    size_t m_nbeta = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;

    /// 1-electron memory map
    mem_map_1 m_map_a;
    mem_map_1 m_map_b;
    /// 2-electron memory maps
    mem_map_2 m_map_aa;
    mem_map_2 m_map_ab;
    mem_map_2 m_map_bb;
    // Get access to memory maps
    mem_map_1 &get_map(bool alpha) 
    { 
        return alpha ? m_map_a : m_map_b; 
    }
    mem_map_2 &get_map(bool alpha1, bool alpha2) 
    { 
        return alpha1 ? (alpha2 ? m_map_aa : m_map_ab) : m_map_bb; 
    }

    /// Determinant list
    std::map<Determinant,int> m_dets;

    /// Compute scalar part of sigma vector
    void H0_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma);
    /// Compute the one-electron part of the sigma vector
    void H1_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha);
    /// Compute the two-electron part of the sigma vector
    void H2_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha1, bool alpha2);

    /// Build the CI space
    void initialize(std::string citype);
    /// Build FCI determinants
    void build_fci_determinants();
    /// Build CIS determinants
    void build_cis_determinants(bool with_ref);
    /// Build memory maps
    void build_memory_map1(bool alpha);
    void build_memory_map2(bool alpha1, bool alpha2);

    /// Build the Hamiltonian matrix
    void build_H0(std::vector<double> &Hmat);
    void build_H1(std::vector<double> &Hmat, bool alpha);
    void build_H2(std::vector<double> &Hmat, bool alpha1, bool alpha2);
};

#endif // CI_SPACE_H