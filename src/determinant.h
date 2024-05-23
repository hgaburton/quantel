#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <cassert>
#include <algorithm>
#include <vector>
#include <string>
#include <tuple>
#include "excitation.h"

class Determinant {
public:
    /// Default destructor
    virtual ~Determinant() { }

    /// Default constructor
    Determinant() { }

    /// Constructor with occupation vectors
    Determinant(std::vector<uint8_t> occ_alfa, std::vector<uint8_t> occ_beta) :
        m_occ_alfa(occ_alfa), m_occ_beta(occ_beta)
    { 
        // Check dimensions
        assert(m_occ_alfa.size() == m_occ_beta.size());
        m_nmo = m_occ_alfa.size();
    }

    /// Comparison operator
    bool operator< (const Determinant &rhs) const 
    {
        if(m_occ_alfa > rhs.m_occ_alfa) return true;
        if(m_occ_alfa < rhs.m_occ_alfa) return false;
        if(m_occ_beta > rhs.m_occ_beta) return true;
        return false;
    }

    /// Apply single excitation operator to the determinant
    /// \param Epq Excitation operator
    /// \param alpha True if alpha excitation, false if beta
    int apply_excitation(Eph &Epq, bool alpha);

    /// Apply double excitation operator to the determinant
    /// \param Eqp First excitation operator
    /// \param Esr Second excitation operator
    /// \param alpha1 Spin of the first excitation
    /// \param alpha2 Spin of the second excitation
    int apply_excitation(Epphh &Epqrs, bool alpha1, bool alpha2);

    /// Alfa occupation vector
    std::vector<uint8_t> m_occ_alfa;
    /// Beta occupation vector
    std::vector<uint8_t> m_occ_beta;

    /// Number of orbitals
    size_t m_nmo = 0;
};

/// @brief Print a determinant
std::string det_str(const Determinant &det);

#endif // DETERMINANT_H