#ifndef CONFIG_MATRIX_CALCULATOR_H
#define CONFIG_MATRIX_CALCULATOR_H

#include "configuration.h"
#include <armadillo> 


class MatrixElementCalculator { 
public: 
    // Default Constructor 
    MatrixElementCalculator() {} 

    double one_body_coupling( const Configuration &bra, const Configuration &ket, const uint8_t i, const uint8_t j  ) const ; 
    double two_body_coupling( const Configuration &bra, const Configuration &ket, const uint8_t i, const uint8_t j , const uint8_t k, const uint8_t l ) const ;
    double one_body_fragment(const int &level, const int &d1, const int &d2, const int &b, const int &delta_b, const int &head, const int &tail, const int &RorL ) const ; 

    // Helper functions 
    bool contains(const std::vector<std::string>& vec, const std::string& val) const; 
    std::string HeadsOrTails(const int &ind, const int &i, const int &j) const ; 
    int get_Dind(const int &delta_b) const ; 

};

#endif