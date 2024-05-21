#ifndef CI_EXPANSION_H
#define CI_EXPANSION_H

#include <vector>
#include <map>
#include "determinant.h"
#include "mo_integrals.h"
#include "excitation.h"
#include "ci_space.h"

class CIexpansion {
/// \brief This class is used to represent an arbitrary CI wavefunction
public:
    /// Default destructor
    virtual ~CIexpansion() { }

    /// Initialise from MO integrals
    CIexpansion(MOintegrals &mo_ints, CIspace &hilbert_space) : 
        m_ints(mo_ints), m_nmo(mo_ints.nmo()),
        m_hilbert_space(hilbert_space), m_ndet(hilbert_space.ndet())
    { 
      assert(mo_ints.nmo() == hilbert_space.nmo());
    }

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
    size_t m_nmo = 0;
    /// CI space
    CIspace &m_hilbert_space;
    size_t m_ndet = 0;
};


#endif // CI_EXPANSION_H