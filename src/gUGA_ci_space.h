#ifndef GUGA_CI_SPACE_H
#define GUGA_CI_SPACE_H

#include <cstddef>
#include <map>
#include <vector>
#include <tuple>
#include <iostream>
#include <string>
#include <cassert>
#include "excitation.h"
#include "configuration.h"
#include "mo_integrals.h"
#include "drt.h"
#include "fmt/core.h"

// Excitation maps organised with {excitation} = < ket index, bra index, coeff > 
typedef std::map<Eph, std::vector<std::tuple<size_t,size_t,double> > > config_mem_map_1 ; 
typedef std::map<Epphh, std::vector<std::tuple<size_t,size_t,double> > > config_mem_map_2 ; 
class GUGA_CIspace{
public: 
    // Default destructor 
    virtual ~GUGA_CIspace() {}
    
    // Constructor 
    GUGA_CIspace( MOintegrals &mo_ints, size_t nmo, size_t nelec, double totspin ): 
    m_ints(mo_ints), m_nmo(nmo), m_nelec(nelec), m_totspin(totspin), m_drtobj(nmo,nelec,totspin) 
    { 
        if(m_nmo < 0)
            throw std::runtime_error("GUGA_CIspace::CIspace: Negative number of active orbitals");
        if(m_nelec > 2*m_nmo)
            throw std::runtime_error("GUGA_CIspace::CIspace: Invalid number of electrons");
        if(fabs(m_totspin) > 0.5 * m_nelec)
            throw std::runtime_error("GUGA_CIspace::CIspace: Invalid Total spin ");
    }
    
    // Build CI space 
    void initialize(std::string citype, std::vector<std::string> configlist={}); 
    
    // Print configuration list 
    virtual void print() const; 
    // Print CI vector
    virtual void print_vector(const std::vector<double> &ci_vec, double tol) const; 
    
    size_t nconfigs() const { return m_nconfigs; }
    size_t nmo() const { return m_nmo; }

    std::vector<Configuration> get_basis() ; 

    int get_config_index( const Configuration &config ) const { 
        auto it = m_configs.find(config);
        if(it == m_configs.end()) { 
            throw std::runtime_error("GUGA_CIspace::get_config_index Configuration not found");
        }
        return it->second; 
    }

    std::vector<std::string> get_config_list() const { 
        // Get list of tuples so we can sort by index
        std::vector<std::tuple<int,std::string>> configs;
        for(auto &[config, ind] : m_configs) {
            configs.push_back(std::make_tuple(ind,config.config_str()));
        } 
        // Sort the list by index
        std::sort(configs.begin(), configs.end(), [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        
        // Convert to a list of strings without the index
        std::vector<std::string> configs_str;
        for(auto &[ind, configstr] : configs)
            configs_str.push_back(configstr);
        return configs_str;
    }

    std::vector<size_t> check_map(Configuration &ket, Eph &Epq) ;  
    std::vector<size_t> check_map(Configuration &ket, Epphh &Epqrs) ; 

    void H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma) ; 
    void build_Hd(std::vector<double> &Hdiag);
    void resolve_build_Hd(std::vector<double> &Hdiag);

    void print_map1(Eph &Epq) ;
    DRT m_drtobj ; 
    // 1-electron excitation memory map 
    config_mem_map_1 m_map1 ;     
    // 2-electron excitation memory map 
    config_mem_map_2 m_map2 ;     

    double resolve_two_body_matrix_element( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs ) const ;
    void print_map_couplings(Configuration &bra, Configuration &ket) ; 
    
    std::vector<std::tuple<Eph,double>> map1_couplings(Configuration &bra, Configuration &ket) ;
    std::vector<std::tuple<Epphh,double>> map2_couplings(Configuration &bra, Configuration &ket) ;

    void build_Hmat(std::vector<double> &Hmat) ; 
    void build_H0(std::vector<double> &H0) ;
    void build_H1(std::vector<double> &H1) ;
    void build_H2(std::vector<double> &H2) ; 
    
    void build_fci_configs(); 
    private: 
    /// MO Integrals 
    MOintegrals &m_ints; 
    bool m_initialized = false ; 
    size_t m_nelec;
    size_t m_nmo ;
    double m_totspin ;
    size_t m_nconfigs = 0 ; 
   

    // Configuration list 
    std::map<Configuration,int> m_configs; 
    void build_memory_map1(); 
    void build_memory_map2(); 
    void resolve_build_memory_map2(); 
}; 

#endif // GUGA_CI_SPACE_H 
