#include <omp.h>
#include <algorithm>
#include "ci_expansion.h"
#include "fmt/format.h"
#include "excitation.h"

void CIexpansion::sigma_vector(std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    // Check size of input
    assert(ci_vec.size() == m_ndet);
    // Resize output
    sigma.resize(m_ndet,0.0);

    // Compute scalar part of sigma vector
    sigma_scalar(ci_vec, sigma);
    // Get one-electron part of sigma vector
    sigma_one_electron(ci_vec, sigma, true);
    sigma_one_electron(ci_vec, sigma, false);
    // Get one-electron part of sigma vector
    sigma_two_electron(ci_vec, sigma, true, false);
    sigma_two_electron(ci_vec, sigma, true, true);
    sigma_two_electron(ci_vec, sigma, false, false);
}

void CIexpansion::sigma_scalar(std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    double v_scalar = m_ints.scalar_potential();
    for(size_t ind=0; ind<m_ndet; ind++)
        sigma[ind] += ci_vec[ind] * v_scalar;
}

void CIexpansion::sigma_one_electron(std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha)
{
    // Get one-electron integrals
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Pair key and connections
        Excitation key = {p,q,alpha};
        auto connect = m_hilbert_space.m_map1[key];
        // Get one-electron integral
        double hpq = m_ints.oei(p,q,alpha);
        // Loop over connections
        for(auto &[indJ, indI, phase] : connect)
            sigma[indI] += phase * hpq * ci_vec[indJ];
    }
}

void CIexpansion::sigma_two_electron(
    std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha1, bool alpha2)
{
    assert(alpha1 >= alpha2);
    double tol = 1e-14;

    // Scaling factor for same spin terms is 1/4 due to antisymmetrisation
    double scale = (alpha1 == alpha2) ? 0.25 : 1.0;

    // Account for effective two-electron integrals
    if(alpha1 == alpha2) 
    {
        // Get one-electron integrals
        for(size_t p=0; p<m_nmo; p++)
        for(size_t q=0; q<m_nmo; q++)
        {
            // Pair key and connections
            Excitation key = {p,q,alpha1};
            auto connect = m_hilbert_space.m_map1[key];
            // Get one-electron integral
            double Kpq = scale * m_ints.oek(p,q,alpha1);
            // Loop over connections
            for(auto &[indJ, indI, phase] : connect)
                sigma[indI] -= phase * Kpq * ci_vec[indJ];
        }
    }

    /*
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    {
        // Second excitation key
        Excitation key1 = {p,r,alpha1};
        for(size_t q=0; q<m_nmo; q++)
        for(size_t s=0; s<m_nmo; s++)
        {
            // Get two-electron integral
            double vpqrs = scale * m_ints.tei(p,q,r,s,alpha1,alpha2);
            if(std::abs(vpqrs) < tol) continue;
            // First excitation key
            Excitation key2 = {q,s,alpha2};
            auto connect = m_hilbert_space.m_map2[std::make_tuple(key1,key2)];
            // Loop over connections
            for(auto &[indJ, indI, phase] : connect)
                sigma[indI] += phase * vpqrs * ci_vec[indJ];
        }
    }
    */
    
    // Implementation with D vector method
    // Build D vector
    std::vector<double> Dqs(m_nmo*m_nmo*m_ndet,0.0);
    #pragma omp parallel for collapse(2)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Data buffer 
        double *buff = &Dqs[q*m_nmo*m_ndet+s*m_ndet];
        // Pair key and connections
        Excitation key = {q,s,alpha2};
        auto connections = m_hilbert_space.m_map1[key];
        // Loop over connections
        for(auto &[indJ, indK, phase] : connections)
            buff[indK] += scale * phase * ci_vec[indJ];
    }

    // Transform D vector
    std::vector<double> Dpr(m_nmo*m_nmo*m_ndet,0.0);
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    {
        // Data buffer 
        double *buff = &Dpr[p*m_nmo*m_ndet+r*m_ndet];
        for(size_t ind=0; ind<m_ndet; ind++)
        for(size_t q=0; q<m_nmo; q++)
        for(size_t s=0; s<m_nmo; s++)
        {
            // Get two-electron integral
            double vpqrs = m_ints.tei(p,q,r,s,alpha1,alpha2);
            buff[ind] += vpqrs * Dqs[q*m_nmo*m_ndet+s*m_ndet+ind];
        }
    }

    // Compute sigma vector
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    {
        // Data buffer 
        double *buff = &Dpr[p*m_nmo*m_ndet+r*m_ndet];
        // Pair key and connected determinants
        Excitation key = {p,r,alpha1};
        auto connect = m_hilbert_space.m_map1[key];
        for(auto &[indK, indI, phase] : connect)
            sigma[indI] += phase * buff[indK];
    }
}