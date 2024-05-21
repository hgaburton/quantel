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
    CIexpansion(MOintegrals &mo_ints, size_t nalfa, size_t nbeta) : 
        m_ints(mo_ints), m_nmo(mo_ints.nmo()), m_nalfa(nalfa), m_nbeta(nbeta) 
    { }

    /// Define CI space from a list of determinants
    void build_space();

    /// Print the determinant list
    void print();

    /// Print a CI vector
    void print_vector(std::vector<double> &ci_vec);

    /// Compute the sigma vector
    void sigma_vector(std::vector<double> &ci_vec, std::vector<double> &sigma);

    /// Compute scalar part of sigma vector
    void sigma_scalar(std::vector<double> &ci_vec, std::vector<double> &sigma);
    /// Compute the one-electron part of the sigma vector
    void sigma_one_electron(std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha);
    /// Compute the two-electron part of the sigma vector
    void sigma_two_electron(std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha1, bool alpha2);

    /// Get the number of determinants
    size_t ndet() const { return m_ndet; }

private:
    /// MO integrals
    MOintegrals &m_ints;

    /// Variational determinant list, maps determinant to index
    std::map<Determinant,int> m_dets;

    /// @brief Map of determinants to connected determinants
    std::map<std::tuple<size_t,size_t,bool>, std::vector<std::tuple<size_t,size_t,int> > > m_map;
    
    /// Number of determinants
    size_t m_ndet = 0;  
    size_t m_ndeta = 0;
    size_t m_ndetb = 0;
    /// Number of auxiliary determinants
    size_t m_det_aux = 0;
    /// Number of molecular orbitals
    size_t m_nmo = 0;
    /// Number of alpha electrons
    size_t m_nalfa = 0;
    /// Number of beta electrons
    size_t m_nbeta = 0;

};


#endif // CI_EXPANSION_H