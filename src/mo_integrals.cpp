#include "mo_integrals.h"
#include "linalg.h"

void MOintegrals::update_orbitals(
    std::vector<double> C, size_t ncore, size_t nactive)
{
    /// Check we have a valid number of orbitals
    if(ncore + nactive > m_nmo)
        throw std::runtime_error("MOintegrals::compute_integrals: Invalid number of orbitals");

    /// Save the number of correlated orbitals
    m_ncore = ncore;
    m_nact = nactive;

    /// Check the dimensions
    if(C.size() != m_nbsf * m_nmo)
        throw std::runtime_error("MOintegrals::compute_integrals: Orbital coefficients have wrong dimensions");

    /// Save the orbital coefficients
    m_C = C;

    /// Save the active orbital coefficients
    m_Cact.resize(m_nbsf*m_nact,0.0);
    #pragma omp parallel for collapse(2)
    for(size_t mu=0; mu<m_nbsf; mu++)
    for(size_t p=0; p<m_nact; p++)
        m_Cact[mu*m_nact+p] = C[mu*m_nmo+(p+m_ncore)];

    /// Compute the inactive Fock matrix
    compute_core_potential();

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

void MOintegrals::compute_core_density()
{
    m_Pcore.resize(m_nbsf*m_nbsf,0.0);

    #pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nbsf; p++)
    for(size_t q=0; q<m_nbsf; q++)
    {
        for(size_t i=0; i<m_ncore; i++)
            m_Pcore[p*m_nbsf+q] += m_C[p*m_nmo+i] * m_C[q*m_nmo+i];
    }
}

void MOintegrals::compute_core_potential()
{
    // Resize core potential in active orbital basis
    m_Vc_oei.resize(m_nact*m_nact,0.0);
    m_Vc = 0;

    if(m_ncore > 0)
    {
        /// Compute the core density matrix
        compute_core_density();

        // Compute inactive JK matrix (2J-K) in AO basis
        std::vector<double> JK(m_nbsf*m_nbsf,0.0);
        m_ints.build_JK(m_Pcore,JK);

        // Compute scalar core energy
        double *Hao = m_ints.oei_matrix(true);
        #pragma omp parallel for reduction(+:m_Vc)
        for(size_t pq=0; pq<m_nbsf*m_nbsf; pq++)
            m_Vc += 2 * (Hao[pq] + 0.5 * JK[pq]) * m_Pcore[pq];

        // Transform to active orbital basis to give core one-electron potential
        oei_transform(m_Cact,m_Cact,JK,m_Vc_oei,m_nact,m_nact,m_nbsf);
    }
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
    if(alpha1 == false and alpha2 == true)
        return m_tei_ab[tei_index(q,p,s,r)];
    return 0;
}

void MOintegrals::compute_scalar_potential()
{
    m_V = m_ints.scalar_potential() + m_Vc;
}

void MOintegrals::compute_oei(bool alpha)
{
    /// Get relevant vectors
    std::vector<double> &v_C = m_Cact;
    std::vector<double> &v_oei = alpha ? m_oei_a : m_oei_b;
    // Compute 1-electron transformation
    m_ints.oei_ao_to_mo(v_C,v_C,v_oei,alpha);
    // Add inactive Fock component
    if(m_ncore > 0)
    {
        for(size_t pq=0; pq<m_nact*m_nact; pq++)
            v_oei[pq] += m_Vc_oei[pq];
    }
}

void MOintegrals::compute_tei(bool alpha1, bool alpha2)
{
    // Only allow aa, ab, bb
    assert(alpha1 >= alpha2);
    /// Get relevant vectors
    std::vector<double> &v_C1 = m_Cact;
    std::vector<double> &v_C2 = m_Cact;
    std::vector<double> &v_tei = alpha1 ? (alpha2 ? m_tei_aa : m_tei_ab) : m_tei_bb;
    // Compute transformation
    m_ints.tei_ao_to_mo(v_C1,v_C2,v_C1,v_C2,v_tei,alpha1,alpha2);
}