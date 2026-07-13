#ifndef GUGA_EVALUATOR_H
#define GUGA_EVALUATOR_H

#include "configuration.h"
#include <armadillo> 
#include "excitation.h"


class GUGAEval { 
public: 
    // Default Constructor 
    GUGAEval() {} 

    // Calculating single elements       
    double one_body_coupling( const Configuration &bra, const Configuration &ket, const Eph &Eph  ) const ; 
    double two_body_coupling( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs ) const ;
    double one_body_fragment(const int &level, const int &d1, const int &d2, const int &b, const int &delta_b, const int &head, const int &tail, const int &RorL ) const ;
    double resolve_two_body_matrix_element( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs, const std::vector<Configuration> &basis) const ; 
    //std::vector<int> one_body_drt_step(int &ref_step, int &level, Eph &Epq) ; 
    //std::vector<int> two_body_drt_step(int &ref_step, int &level, Epphh &Epqrs) ; 

    // Helper functions 
    bool contains(const std::vector<std::string>& vec, const std::string& val) const; 
    std::string HeadsOrTails(const int &ind, const int &i, const int &j) const ; 
    int get_Dind(const int &delta_b) const ; 

};

#endif //GUGA_EVALUATOR_H
