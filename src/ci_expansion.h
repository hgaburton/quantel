#ifndef CI_EXPANSION_H
#define CI_EXPANSION_H

#include <vector>
#include <map>
#include "determinant.h"
#include "mo_integrals.h"

class CIexpansion {
/// \brief This class is used to represent an arbitrary CI wavefunction
public:
    /// Default destructor
    virtual ~CIexpansion() { }

    /// Initialise from MO integrals
    CIexpansion(MOintegrals &mo_ints) : 
        m_ints(mo_ints), m_ndet(0), m_nmo(mo_ints.nmo()) 
    { }

    /// Define CI space from a list of determinants
    void define_space(std::vector<Determinant> det_list);

    /// Print the determinant list
    void print();

    /// Print a CI vector
    void print_vector(std::vector<double> &ci_vec);

    /// Compute the sigma vector
    void sigma_vector(std::vector<double> &ci_vec, std::vector<double> &sigma);

    /// Compute the one-electron part of the sigma vector
    void sigma_one_electron(std::vector<double> &ci_vec, std::vector<double> &sigma);
    /// Compute the two-electron part of the sigma vector
    void sigma_two_electron(std::vector<double> &ci_vec, std::vector<double> &sigma);

    /// Get the number of determinants
    size_t ndet() const { return m_ndet; }

private:
    /// MO integrals
    MOintegrals &m_ints;

    /// Determinant list, maps determinant to index
    std::map<Determinant, size_t> m_dets;
    /// Auxiliary determinant list, containing connected determinants
    std::map<Determinant, size_t> m_aux_dets;
    
    /// Number of determinants
    size_t m_ndet = 0;  
    /// Number of auxiliary determinants
    size_t m_det_aux = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;

};


#endif // CI_EXPANSION_H