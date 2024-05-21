#ifndef CI_EXPANSION_H
#define CI_EXPANSION_H

#include <vector>
#include <map>
#include "determinant.h"


class CIexpansion {
/// \brief This class is used to represent an arbitrary CI wavefunction
public:
    /// Default destructor
    virtual ~CIexpansion() { }

    /// Default constructor
    CIexpansion() { }

    /// Initialise from a list of determinants
    CIexpansion(std::vector<Determinant> det_list);

    /// Initialise from a list of determinants and coefficients
    CIexpansion(std::vector<Determinant> det_list, std::vector<double> coeff);

    /// Print the CI vector
    void print(double thresh=1e-6);

private:
    /// Determinant list
    std::map<Determinant, double> m_dets;
    /// 
    /// Number of determinants
    size_t m_ndet = 0;  

};


#endif // CI_EXPANSION_H