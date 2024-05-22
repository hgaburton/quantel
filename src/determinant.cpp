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
    const size_t &p = Epqrs.particle1; // Particle 1 index
    const size_t &q = Epqrs.particle2; // Particle 2 index
    const size_t &r = Epqrs.hole1; // Hole 1 index
    const size_t &s = Epqrs.hole2; // Hole 2 index
    
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

    // Apply first operator
    if(occ1[r]==0) return 0;
    occ1[r] = 0;
    for(size_t i=r+1; i<m_nmo; i++) 
        phase_exp += occ1[i];

    // Apply second operator
    if(occ2[s]==0) return 0;
    occ2[s] = 0;
    for(size_t i=s+1; i<m_nmo; i++) 
        phase_exp += occ2[i];
    
    // Apply third operator
    if(occ2[q]==1) return 0;
    occ2[q] = 1;
    for(size_t i=q+1; i<m_nmo; i++) 
        phase_exp += occ2[i];

    // Apply fourth operator
    if(occ1[p]==1) return 0;
    occ1[p] = 1;
    for(size_t i=p+1; i<m_nmo; i++) 
        phase_exp += occ1[i];

    // Get the phase
    if(phase_exp % 2 == 1) 
        return -1;
    else
        return 1;
}