#ifndef DRT_H
#define DRT_H

#include <string>
#include <algorithm>
#include <vector> 
#include <armadillo> 
#include "excitation.h"
#include "configuration.h"

class DRT{ 
    public: 

        // Default destructor 
        virtual ~DRT(){}
        // Default Constructor 
        DRT() {}
        
        // Constructor with arguments 
        DRT( size_t nmo, size_t nelec, double totspin ) {
            m_a = (int) ( nelec - 2*totspin)/2 ; 
            m_b = (int) 2 * totspin ; 
            m_c = (int) nmo - m_a - m_b;
            
            construct_drt() ; 
        }

        void construct_drt() ; 
        arma::uvec one_body_step(int &ref_step, int &level, Eph &Epq) ; 
        arma::uvec two_body_step(int &ref_step, int &level, Epphh &Epqrs) ; 
        std::vector< Configuration > drt_loop( Configuration &ref_config, int &head, int &tail, std::function<arma::uvec(int&, int&)> stepfunc) ; 

        // One body and two body branching, because we want DRT to know about Configuration but probably necessary for it to know in the other direction 
        std::vector<std::tuple<Configuration,double>> apply_excitation(Configuration &ket,  Eph &Epq) ;
        std::vector<std::tuple<Configuration,double>> apply_excitation(Configuration &ket,  Epphh &Epqrs) ;
    
        arma::imat m_drt ;   
    private: 
        int m_a = 0 ;
        int m_b = 0 ; 
        int m_c = 0 ;
};

#endif // DRT_H  