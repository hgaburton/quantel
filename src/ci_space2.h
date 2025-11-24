#ifndef CI_SPACE_2_H
#define CI_SPACE_2_H

#include <cstddef>
#include <map>
#include <vector>
#include <tuple>
#include <iostream>
#include <string>
#include <cassert>
#include "determinant.h"
#include "excitation.h"
#include "mo_integrals.h"
#include "fmt/core.h"

typedef std::tuple<Eph,bool> exc;
typedef std::tuple<size_t,size_t,int> det_map;

/// \brief Container for information about a CI space
class CIspace2 
{
public:
    /// Destructor
    virtual ~CIspace2() { }

    /// Constructor
    CIspace2(MOintegrals &mo_ints, size_t norb, size_t nalfa, size_t nbeta) :
        m_ints(mo_ints), m_nmo(norb), m_nalfa(nalfa), m_nbeta(nbeta)
    { 
        if(m_nmo < 0)
            throw std::runtime_error("CIspace2::CIspace2: Negative number of active orbitals");
        if(m_nalfa > m_nmo)
            throw std::runtime_error("CIspace2::CIspace2: Invalid number of alpha electrons");
        if(m_nbeta > m_nmo)
            throw std::runtime_error("CIspace2::CIspace2: Invalid number of beta electrons");
    }

    /// Build the CI space
    void initialize(std::string citype, std::vector<std::string> detlist={});

    /// Print the determinant list
    virtual void print() const;
    /// Print the auxiliary determinant list
    virtual void print_auxiliary() const;
    /// Print a CI vector
    virtual void print_vector(const std::vector<double> &ci_vec, double tol) const;

    /// Get the number of determinants
    size_t ndet() const { return m_ndet; }
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
    /// @param Hmat Output Hamiltonian matrix
    void build_Hmat(std::vector<double> &Hmat);

    /// @brief Build the diagonal of the Hamiltonian matrix
    /// @param Hdiag
    void build_Hd(std::vector<double> &Hdiag);

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

    /// @brief Get the index of a determinant
    /// @param det Determinant to get the index of
    /// @return Index of the determinant
    int get_det_index(const Determinant &det) const
    {
        auto it = m_dets.find(det);
        if(it == m_dets.end())
            throw std::runtime_error("CIspace2::get_index: Determinant not found");
        return it->second;
    }

    /// @brief Get a list of tupes with determinants and their index
    /// @return List of determinants
    std::vector<std::string> get_det_list() const
    {
        // Get list of tuples so we can sort by index
        std::vector<std::tuple<int,std::string> > dets;
        for(auto &[det, ind] : m_dets)
            dets.push_back(std::make_tuple(ind,det_str(det)));
        // Sort the list by index
        std::sort(dets.begin(), dets.end(), [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        // Convert to a list of strings without the index
        std::vector<std::string> dets_str;
        for(auto &[ind, det] : dets)
            dets_str.push_back(det);
        return dets_str;
    }

private:
    /// MO integrals
    MOintegrals &m_ints;
    /// Control variable if succesfully initialized
    bool m_initialized = false;

    /// Number of determinants
    size_t m_ndet = 0;
    /// Number of auxiliary determinants
    size_t m_ndet_aux = 0;
    /// Number of electrons
    size_t m_nalfa = 0;
    size_t m_nbeta = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;
    /// Max memory
    size_t m_max_mem = 1L * 1024L * 1024L * 1024L; // 1 GB

    /// Diagonal of the Hamiltonian matrix
    std::vector<double> m_Hd;

    /// Map from excitation to list of (bra_index, ket_index, phase)
    /// Use to find which determinants are connected by a given excitation
    std::map<exc, std::vector<det_map> > m_ex_det;

    /// List from determinant index to map of excitations to (det_index, phase)
    /// Use to find excitations that act on a given determinant
    std::vector<std::map<exc, std::tuple<size_t,int> > > m_det_ex;
    /// Determinant list in CI space
    std::map<Determinant,int> m_dets;

    /// Build FCI determinants
    void build_fci_determinants();
    /// Build CIS determinants
    void build_cis_determinants(bool with_ref);
    /// Build custom determinants
    void build_custom_determinants(std::vector<std::string> detlist);
    /// Build auxiliary determinants for 1-electron excitations
    void build_auxiliary_determinants();
    /// Build memory maps
    void build_memory_map();

};

#endif // CI_SPACE_2_H
