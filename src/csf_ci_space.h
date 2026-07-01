#ifndef CSF_CI_SPACE_H
#define CSF_CI_SPACE_H

#include <string> 
#include <vector> 
#include <armadillo> 
#include <algorithm>
#include <cassert> 
#include <map>
#include "configuration.h"
#include "mo_integrals.h"
#include "excitation.h"
#include "drt.h"


typedef std::map<Eph, std::vector<std::tuple<size_t,size_t,double> > > config_mem_map_1 ; 
class CSF_CIspace{
public: 
    // Default destructor 
    virtual ~CSF_CIspace() {}
    //Default Constructor 
    CSF_CIspace() {} 


    //CSF_CI_SPACE( MOINTEGRALs &mo_ints, size_t norb, size_t nelec, double totspin ): 
    //    m_ints(mo_ints), m_nmo(norb), m_nelec(nelec), m_totspin(totspin) 
    CSF_CIspace( size_t nmo, size_t nelec, double totspin ): 
        m_nmo(nmo), m_nelec(nelec), m_totspin(totspin), m_drtobj(nmo,nelec,totspin) 
    { 
        if(m_nmo < 0)
            throw std::runtime_error("CIspace::CIspace: Negative number of active orbitals");
        if(m_nelec > 2*m_nmo)
            throw std::runtime_error("CIspace::CIspace: Invalid number of electrons");
        if(fabs(m_totspin) > 0.5 * m_nelec)
            throw std::runtime_error("CIspace::CIspace: Invalid Total spin ");

    }
    
    // Build CI space 
    //void initialize(std::string citype, std::vector<std::string> configlist={}); 
    // Print configuration list 
    //virtual void print() const; 
    // Print CI vector
    //virtual void print_vector(const std::vector<double> &ci_vec, double tol) const; 
    //void construct_drt();

    void build_fci_configs(); 

    int get_config_index( const Configuration &config ) const { 
        auto it = m_configs.find(config);
        if(it == m_configs.end()) { 
            throw std::runtime_error("CSF_CIspace::get_config_index Configuration not found");
        }
        return it->second; 
    }

    std::vector<std::vector<uint8_t>> get_config_list() const { 
        // Get list of tuples so we can sort by index
        std::vector<std::tuple<int,std::vector<uint8_t>>> configs;
        for(auto &[config, ind] : m_configs) {
            configs.push_back(std::make_tuple(ind,config.get_vec()));
        } 
        // Sort the list by index
        std::sort(configs.begin(), configs.end(), [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        // Convert to a list of strings without the index
        std::vector<std::vector<uint8_t>> configlist;
        for(auto &[ind, config] : configs)
            configlist.push_back(config);
        return configlist;
    }

    std::vector<Configuration> get_basis() ; 

    DRT m_drtobj ; 
private: 
    /// MO Integrals 
    //MOintegrals &m_ints; 
    //bool m_initialised = false ; 
    //size_t m_config = 0; 
    size_t m_nelec;
    size_t m_nmo ;
    double m_totspin ;
    size_t m_nconfigs = 0 ; 
   
    // 1-electron memory map 
    config_mem_map_1 m_map1 ;     
    //mem_map_2 m_map2 ;     

    // Configuration list 
    std::map<Configuration,int> m_configs; 
    //void build_memory_map1(); 
}; 

#endif // CSF_CI_SPACE_H 