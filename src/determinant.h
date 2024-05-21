#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <cassert>
#include <algorithm>
#include <vector>
#include <string>
#include <tuple>

class Determinant {
public:
    /// Default destructor
    virtual ~Determinant() { }

    /// Default constructor
    Determinant() { }

    /// Constructor with occupation vectors
    Determinant(std::vector<bool> occ_alfa, std::vector<bool> occ_beta) :
        m_occ_alfa(occ_alfa), m_occ_beta(occ_beta)
    { 
        // Check dimensions
        assert(m_occ_alfa.size() == m_occ_beta.size());
        m_nmo = m_occ_alfa.size();
    }

    /// Comparison operator
    bool operator< (const Determinant &rhs) const
    {
        return bitstring() > rhs.bitstring();
    }

    /// Return a string representation of the determinant
    std::string str() const;

    /// Return a bitstring representation of the determinant (alfa then beta)
    std::string bitstring() const;

    /// Return the number of orbitals
    size_t nmo() const { return m_nmo; }

    /// Return the number of electrons
    size_t nelec() const { return nalfa() + nbeta(); };
    /// Return the number of alpha electrons
    size_t nalfa() const
        { return std::count(m_occ_alfa.begin(), m_occ_alfa.end(), true); };
    /// Return the number of beta electrons
    size_t nbeta() const
        { return std::count(m_occ_beta.begin(), m_occ_beta.end(), true); };

    /// Apply single excitation operator to the determinant
    /// \param excitation Tuple containing the indices of the excitation
    /// \param alpha Boolean flag indicating alpha or beta excitation
    std::tuple<Determinant, int> get_excitation(std::tuple<int,int> excitation, bool alpha) const;

private:
    /// Alfa occupation vector
    std::vector<bool> m_occ_alfa;
    /// Beta occupation vector
    std::vector<bool> m_occ_beta;
    /// Number of orbitals
    size_t m_nmo = 0;
    
};

#endif // DETERMINANT_H