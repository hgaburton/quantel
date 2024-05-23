#include "determinant.h"
#include <iostream>

std::string det_str(const Determinant &det)
{ 
    std::string outstr = "";
    for(size_t i = 0; i < det.m_nmo; i++) 
    {
        if(det.m_occ_alfa[i]==1 and det.m_occ_beta[i]==1) 
            outstr += "2";
        else if(det.m_occ_alfa[i]==1) 
            outstr += "a";
        else if(det.m_occ_beta[i]==1) 
            outstr += "b";
        else 
            outstr += "0";
    }
    return outstr;
}

int Determinant::apply_excitation(Eph &Eqp, bool alpha)
{
    // Get the indices of the excitation
    const size_t &p = Eqp.hole; // Hole index
    const size_t &q = Eqp.particle; // Particle index

    // Check that the indices are valid
    assert(p < m_nmo);
    assert(q < m_nmo);

    // Get relevant occupation vector
    uint8_t *occ = alpha ? m_occ_alfa.data() : m_occ_beta.data();

    if(occ[p]==0) 
        return 0;
    else if(p==q) 
        return 1;
    else if (occ[q] == 1)
        return 0;

    // Number of occupied orbitals between p and q
    int count = 0;
    size_t start = std::min(p,q);
    size_t end   = std::max(p,q);
    for(size_t i=start+1; i<end; i++) 
        count += occ[i];

    // Apply the excitation
    occ[p] = 0;
    occ[q] = 1;

    // Apply the phase
    if(count % 2 == 1) 
        return -1;
    else
        return 1;
}

int Determinant::apply_excitation(
    Epphh &Epqrs, bool alpha1, bool alpha2)
{
    // Get the indices of the excitation
    size_t p = Epqrs.particle1; // Particle 1 index
    size_t q = Epqrs.particle2; // Particle 2 index
    size_t r = Epqrs.hole1; // Hole 1 index
    size_t s = Epqrs.hole2; // Hole 2 index
    
    // Check that the indices are valid
    assert(p < m_nmo);
    assert(q < m_nmo);
    assert(r < m_nmo);
    assert(s < m_nmo);

    // Get relevant occupation vector
    uint8_t *occ1 = alpha1 ? m_occ_alfa.data() : m_occ_beta.data();
    uint8_t *occ2 = alpha2 ? m_occ_alfa.data() : m_occ_beta.data();

    // Operator defined as
    // Epqrs = a_p1^+ a_q2^+ a_s2 a_r1

    // Initialise phase
    size_t phase_exp = 0;
    if(alpha1 == alpha2)
    {
        if(p>q) 
        {   
            std::swap(p,q);
            phase_exp += 1;
        }
        if(r>s) 
        {   
            std::swap(r,s);
            phase_exp += 1;
        }
    }

    // Remove electrons operator
    if(occ1[r]==0) return 0;
    occ1[r] = 0;
    if(occ2[s]==0) return 0;
    occ2[s] = 0;

    // Count number of occupied orbitals between p and r
    size_t start = std::min(p,r);
    size_t end   = std::max(p,r);
    for(size_t i=start+1; i<end; i++) 
        phase_exp += occ1[i];
    // Count number of occupied orbitals between q and s
    start = std::min(q,s);
    end   = std::max(q,s);
    for(size_t i=start+1; i<end; i++) 
        phase_exp += occ2[i];
    // Add electrons back in
    if(occ2[q]==1) return 0;
    occ2[q] = 1;
    if(occ1[p]==1) return 0;
    occ1[p] = 1;

    // Get the phase
    if(phase_exp % 2 == 1) 
        return -1;
    else
        return 1;
}