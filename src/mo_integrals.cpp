#include "mo_integrals.h"

void MOintegrals::initialize()
{
    /// Compute scalar potential
    compute_scalar_potential();
    /// Compute one-electron integrals
    compute_oei(true);
    compute_oei(false);
    /// Compute two-electron integrals
    compute_tei(true,true);
    compute_tei(true,false);
    compute_tei(false,false);
}

double MOintegrals::oei(size_t p, size_t q, bool alpha)
{
    return alpha ? m_oei_a[oei_index(p,q)] : m_oei_b[oei_index(p,q)];
}

double MOintegrals::tei(size_t p, size_t q, size_t r, size_t s, bool alpha1, bool alpha2)
{
    size_t index = tei_index(p,q,r,s);
    if(alpha1 == true and alpha2 == true)
        return m_tei_aa[index];
    if(alpha1 == true and alpha2 == false)
        return m_tei_ab[index];
    if(alpha1 == false and alpha2 == false)    
        return m_tei_bb[index];
    return 0;
}

void MOintegrals::compute_scalar_potential()
{
    m_V = m_ints.scalar_potential();
}

void MOintegrals::compute_oei(bool alpha)
{
    /// Get relevant vectors
    std::vector<double> &v_C = alpha ? m_Ca : m_Cb;
    std::vector<double> &v_oei = alpha ? m_oei_a : m_oei_b;
    // Compute transformation
    m_ints.oei_ao_to_mo(v_C,v_C,v_oei,alpha);
}

void MOintegrals::compute_tei(bool alpha1, bool alpha2)
{
    // Only allow aa, ab, bb
    assert(alpha1 >= alpha2);
    /// Get relevant vectors
    std::vector<double> &v_C1 = alpha1 ? m_Ca : m_Cb;
    std::vector<double> &v_C2 = alpha2 ? m_Ca : m_Cb;
    std::vector<double> &v_tei = alpha1 ? (alpha2 ? m_tei_aa : m_tei_ab) : m_tei_bb;
    // Compute transformation
    m_ints.tei_ao_to_mo(v_C1,v_C2,v_C1,v_C2,v_tei,alpha1,alpha2);
}