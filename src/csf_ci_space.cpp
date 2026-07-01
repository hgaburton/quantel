#include "csf_ci_space.h" 
#include "configuration.h"
#include <armadillo> 
#include <vector>
#include <iostream>
#include "drt.h"


void CSF_CIspace::build_fci_configs() { 
    // Construct DRT and coupling table 
    std::vector< std::tuple< std::vector<uint8_t>, size_t > > configurations = {std::make_tuple( std::vector<uint8_t>{}, m_drtobj.m_drt.n_rows)};
    for ( size_t level = 0 ; level < m_nmo ; level ++ ) {
        std::vector< std::tuple< std::vector<uint8_t>, size_t > > new_configurations;  
        arma::uvec lexical_inds = arma::find(m_drtobj.m_drt.col(3) == (int) level);
        for ( arma::uword ind : lexical_inds ) {
            arma::uvec steps = arma::find(m_drtobj.m_drt.row(ind).subvec(4,m_drtobj.m_drt.n_cols-1) != 0);
            for ( auto configuration : configurations ) {
                if (std::get<1>(configuration)==ind+1){
                    for (auto step : steps){
                        auto temp = std::get<0> (configuration) ;
                        temp.push_back( (uint8_t) step);
                        new_configurations.push_back(std::make_tuple(temp, m_drtobj.m_drt(ind, 4 + (int) step)));
                    }
                } 
            }
        }
        configurations = std::move(new_configurations);
    }
    // Convert to Configuration object and store in map 
    m_nconfigs = 0 ; 
    for (const auto& config : configurations) {
        m_configs[Configuration(std::get<0>(config))] = m_nconfigs++ ; 
    }
    return ; 
}

std::vector<Configuration> CSF_CIspace::get_basis() {
    std::vector<Configuration> basis ; 
    for (auto& [ config, ind] : m_configs) { 
        basis.push_back(config); 
    } 
    return basis; 
}

//void CSF_CIspace::build_memory_map1() { 
//    for (size_t p=0 ; p<m_nmo ; p++){ 
//        for (size_t q=0 ; q<m_nmo ; q++){ 
//            if(q>p) continue; 
//            // Make an excitation 
//            Eph Epq = {p,q}; 
//            Eph Eqp = {q,p};
//            
//            // Initialise map vectors 
//            #pragma omp critical 
//            { 
//                m_map1[Epq] = std::vector<std::tuple<size_t,size_t,double>> (); 
//                if (Epq != Eqp) { 
//                    m_map1[Eqp] = std::vector<std::tuple<size_t,size_t,double>> (); 
//                }
//            }
//            
//            // Loop over configurations 
//            // Does this catch the case in which this is empty?
//            for (auto &[configJ, indJ] : m_configs) { 
//                // Need to loop over the possible excitations
//                Configuration configI = configJ ; 
//                // Compute the R operator excitations 
//                std::vector<std::tuple<Configuration, double>> excitations = configI.apply_excitation(m_drt,Eqp); 
//                for (auto ex : excitations) {
//                    // if this cant be found we move on 
//                    try { 
//                        size_t indI = m_configs.at(std::get<0> (ex)); 
//                        m_map1[Eqp].push_back(std::make_tuple(indJ,indI,std::get<1>(ex))) ; 
//                        if (Eqp != Epq ) { 
//                            // L operator relationships
//                            m_map1[Epq].push_back(std::make_tuple(indI,indJ,std::get<1>(ex))) ; 
//                        }
//                    } 
//                    catch( const std::out_of_range& e) { 
//                        continue; 
//                    }
//                }
//            }
//        }
//    }
//}
