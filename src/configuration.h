#ifndef CONFIGURATION_H 
#define CONFIGURATION_H 

#include <string>
#include <vector> 
#include <armadillo> 
#include <algorithm>

class Configuration { 
public: 
    // Default destructor 
    virtual ~Configuration() { } 
    
    // Default Constructor 
    Configuration() {} 

    // Constructor from step-vector representation 
    Configuration(std::vector<uint8_t> step_vec) 
    {
        // should introduce a check here to ensure drt is valid 
        // i.e. every 2 has been balanced by a 1 before it. 
        // its going to have to iterate through the drt and count the number of ones and two at the same time 
        int count1 = 0;
        int count2 = 0;
        for (int i=0 ; i < step_vec.size() ; i++) {
            if (int(step_vec[i]) == 1) { 
                count1++ ;
            } 
            else if (int(step_vec[i]) == 2) { 
                count2++ ;
            }
            
            if (count2 > count1) {
                throw std::runtime_error("Invalid Step vector: S falls below zero at index " + std::to_string(i) + "."); 
            } 
        } 
        int count3 = std::count(step_vec.begin(), step_vec.end(), 3);
        m_nelec = count1 + count2 + 2*count3 ; 
        m_totspin = 0.5*( count1 - count2) ;
        m_nmo = step_vec.size() ; 
        //std::copy(step_vec.begin(), step_vec.end(), std::back_inserter(m_step_vec) ) ; 
        m_step_vec = step_vec ; 
    }

    // Comparision operator used to override the binaries
    // and so we can order the vectors in some form
    bool operator< (const Configuration &rhs ) const 
    {
        // "Smallest" drt is closed shell wrt specified ordering 
        if (m_step_vec > rhs.m_step_vec ) return true;  
        if (m_step_vec < rhs.m_step_vec ) return false;
        return true;  
    }

    // number of molecular orbitals 
    uint8_t m_nmo; 
    uint8_t m_nelec; 
    double m_totspin;
    std::vector<uint8_t> m_step_vec; 

    // Define functions here 

    // generate Paldus table representation from drt
    // trailing const means function cant modify class variabels
    arma::imat generate_paldus() const;
    // better in s Configuration space object? 
    arma::imat construct_drt() const; 
}; 



#endif // CONFIGURATION_H 
