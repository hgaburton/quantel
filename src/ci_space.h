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
#include "fmt/core.h"

/// \brief Container for information about a CI space
class CIspace 
{
public:
    /// Destructor
    virtual ~CIspace() { }

    /// Constructor
    CIspace(size_t nmo, size_t nalfa, size_t nbeta, std::string citype="FCI") :
        m_nmo(nmo), m_nalfa(nalfa), m_nbeta(nbeta)
    { 
        initialize(citype);
    }

    /// Print the determinant list
    virtual void print() const 
    {
        for(auto &det : m_dets)
            std::cout << det.first.str() << std::endl;
    }

    /// Print a CI vector
    virtual void print_vector(std::vector<double> &ci_vec, double tol) const
    {
        assert(ci_vec.size() == m_ndet);
        for(auto &[det, ind] : m_dets)
        {
            if(std::abs(ci_vec[ind]) > tol) 
                fmt::print("{:>s}: {:>10.6f}\n", det.str(), ci_vec[ind]);;
        }   
    }

    /// Get the number of determinants
    size_t ndet() const { return m_ndet; }
    size_t ndeta() const { return m_ndeta; }
    size_t ndetb() const { return m_ndetb; }
    /// Get the number of electrons
    size_t nalfa() const { return m_nalfa; }
    size_t nbeta() const { return m_nbeta; }
    /// Get the number of molecular orbitals
    size_t nmo() const { return m_nmo; }

    /// 1-electron memory map
    std::map<Excitation, std::vector<std::tuple<size_t,size_t,int> > > m_map1;
    /// 2-electron memory map
    std::map<std::tuple<Excitation,Excitation>, std::vector<std::tuple<size_t,size_t,int> > > m_map2;
    /// Determinant list
    std::map<Determinant,int> m_dets;

private:
    /// Number of determinants
    size_t m_ndet = 0;
    size_t m_ndeta = 0;
    size_t m_ndetb = 0;
    /// Number of electrons
    size_t m_nalfa = 0;
    size_t m_nbeta = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;

    /// Build the FCI space
    void initialize(std::string citype);
    /// Build determinants
    void build_fci_determinants();
    /// Build memory maps
    void build_memory_maps();
};

#endif // CI_SPACE_H