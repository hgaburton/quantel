#ifndef CI_EXPANSION_H
#define CI_EXPANSION_H

#include <vector>
#include <map>
#include "determinant.h"

class ci_expansion {
/// \brief This class is used to represent an arbitrary CI wavefunction
public:
    /// Default destructor
    virtual ~ci_expansion() { }

    /// Default constructor
    ci_expansion() { }

    /// Initialise from a list of determinants
    ci_expansion(std::vector<Determinant> det_list, std::vector<double> coeff)
    { 
        assert(det_list.size() == coeff.size());
        for(size_t i = 0; i < det_list.size(); i++)
            m_determinants[det_list[i]] = coeff[i];

    }


private:
    /// Determinant list
    std::map<Determinant, double> m_determinants;
    /// 
    /// Number of determinants
    size_t m_ndet = 0;  

};


#endif // CI_EXPANSION_H