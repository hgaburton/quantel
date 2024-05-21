#include "determinant.h"

static std::string det_str(const std::vector<bool> &occ_alfa, const std::vector<bool> &occ_beta, const size_t nmo)
{ 
    assert(occ_alfa.size() == nmo); 
    assert(occ_beta.size() == nmo);

    std::string outstr = "";
    for(size_t i = 0; i < nmo; i++) 
    {
        if(occ_alfa[i] and occ_beta[i]) 
            outstr += "2";
        else if(occ_alfa[i]) 
            outstr += "a";
        else if(occ_beta[i]) 
            outstr += "b";
        else 
            outstr += "0";
    }
    return outstr;
}

std::string Determinant::str() const
{
    return det_str(m_occ_alfa, m_occ_beta, m_nmo);
}

std::string Determinant::bitstring() const
{
    std::string outstr = "";
    // Alpha occupation
    for(size_t i = 0; i < m_nmo; i++) 
    {
        if(m_occ_alfa[i]) 
            outstr += "1";
        else 
            outstr += "0";
    }
    // Beta occupation
    for(size_t i = 0; i < m_nmo; i++) 
    {
        if(m_occ_beta[i]) 
            outstr += "1";
        else 
            outstr += "0";
    }
    return outstr;
}

std::tuple<Determinant, int> Determinant::get_excitation(std::tuple<int,int> excitation, bool alpha) const
{
    // Get the indices of the excitation
    int p = std::get<1>(excitation); // Hole index
    int q = std::get<0>(excitation); // Particle index

    // Check that the indices are valid
    assert(p >= 0 and p < m_nmo);
    assert(q >= 0 and q < m_nmo);

    // Get relevant occupation vector
    std::vector<bool> occ = alpha ? m_occ_alfa : m_occ_beta;

    if(not occ[p]) 
        // If hole orbital is unoccupied, return the current determinant with no weight
        return std::make_tuple(Determinant(m_occ_alfa,m_occ_beta), 0);
    else if(p == q) 
        // If the indices are the same, return the current determinant with sign +1
        return std::make_tuple(Determinant(m_occ_alfa,m_occ_beta), 1);
    else if(occ[q])     
        // If the particle orbital is occupied, return the current determinant with no weight
        return std::make_tuple(Determinant(m_occ_alfa,m_occ_beta), 0);

    // Compute phase from number of occupied orbitals between p and q in spin space
    int phase = 1;
    if(q > p) {
        for(size_t i=p+1; i < q; i++) 
            if(occ[i]) phase *= -1; 
    } else {
        for(size_t i=q+1; i < p; i++) 
            if(occ[i]) phase *= -1;
    }

    // Apply the excitation
    occ[p] = false;
    occ[q] = true;

    // Return new determinant
    if(alpha)
        return std::make_tuple(Determinant(occ,m_occ_beta), phase);
    else
        return std::make_tuple(Determinant(m_occ_alfa,occ), phase);
}