#include <omp.h>
#include <algorithm>
#include "ci_expansion.h"
#include "fmt/format.h"

void CIexpansion::build_space()
{ 
    // Populate m_det with FCI space
    m_ndet = 0;
    std::vector<bool> occ_alfa(m_nmo,false);
    std::vector<bool> occ_beta(m_nmo,false);
    std::fill_n(occ_alfa.begin(), m_nalfa, true);
    std::fill_n(occ_beta.begin(), m_nbeta, true);
    do {
        m_ndeta++;
        do {
            m_ndetb++;
            m_dets[Determinant(occ_alfa,occ_beta)] = m_ndet++;
        } while(std::prev_permutation(occ_beta.begin(), occ_beta.end()));
    } while(std::prev_permutation(occ_alfa.begin(), occ_alfa.end()));

    // Populate m_map with connected determinants
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Pair key
        std::tuple<size_t,size_t,bool> key_alfa(p,q,true);
        std::tuple<size_t,size_t,bool> key_beta(p,q,false);
        // Initialise map vectors
        m_map[key_alfa] = std::vector<std::tuple<size_t,size_t,int> >();
        m_map[key_beta] = std::vector<std::tuple<size_t,size_t,int> >();
        // Loop over determinants
        std::tuple<int,int> Epq(p,q);
        for(auto &[detJ, indJ] : m_dets)
        {
            // Get alfa excitation
            auto det_a = detJ.get_excitation(Epq,true);
            if(std::get<1>(det_a) != 0) 
                m_map[key_alfa].push_back(
                    std::make_tuple(indJ,m_dets[std::get<0>(det_a)],std::get<1>(det_a)));
            // Get beta excitation
            auto det_b = detJ.get_excitation(Epq,false);
            if(std::get<1>(det_b) != 0) 
                m_map[key_beta].push_back(
                    std::make_tuple(indJ,m_dets[std::get<0>(det_b)],std::get<1>(det_b)));
        }   
    }
}

void CIexpansion::print()
{
    /*
    std::cout << "Main determinant list" << std::endl;
    for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        std::cout << it->first.str() << std::endl;

    std::cout << "Auxiliary determinant list" << std::endl;
    for(auto it = m_aux_dets.begin(); it != m_aux_dets.end(); it++)
        std::cout << it->first.str() << std::endl;
    */
}

void CIexpansion::print_vector(std::vector<double> &ci_vec)
{
    // Check size of input
    assert(ci_vec.size() == m_ndet);
    // Print vector
    for(auto &[det, ind] : m_dets)
        fmt::print("{:>s}: {:>10.6f}\n", det.str(), ci_vec[ind]);
}

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
        // Pair key
        std::tuple<size_t,size_t,bool> key(p,q,alpha);
        // Get one-electron integral
        double hpq = m_ints.oei(p,q,alpha);
        // Get connected determinants
        auto connections = m_map[key];
        // Loop over connections
        for(auto &[indJ, indI, phase] : connections)
            sigma[indI] += phase * hpq * ci_vec[indJ];
    }
}

void CIexpansion::sigma_two_electron(
    std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha1, bool alpha2)
{
    assert(alpha1 >= alpha2);

    // Scaling factor for same spin terms is 1/4 due to antisymmetrisation
    double scale = (alpha1 == alpha2) ? 0.25 : 1.0;

    // Build D vector
    std::vector<double> Dqs(m_nmo*m_nmo*m_ndet,0.0);
    #pragma omp parallel for collapse(2)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Data buffer 
        double *buff = &Dqs[q*m_nmo*m_ndet+s*m_ndet];
        // Pair key and connections
        std::tuple<size_t,size_t,bool> key(q,s,alpha2);
        auto connections = m_map[key];
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
        std::tuple<size_t,size_t,bool> key(p,r,alpha1);
        auto connect = m_map[key];
        for(auto &[indK, indI, phase] : connect)
            sigma[indI] += phase * buff[indK];
    }

    // Account for effective two-electron integrals
    if(alpha1 == alpha2) 
    {
        // Get one-electron integrals
        for(size_t p=0; p<m_nmo; p++)
        for(size_t q=0; q<m_nmo; q++)
        {
            // Pair key and connections
            std::tuple<size_t,size_t,bool> key(p,q,alpha1);
            auto connect = m_map[key];
            // Get one-electron integral
            double Kpq = scale * m_ints.oek(p,q,alpha1);
            // Loop over connections
            for(auto &[indJ, indI, phase] : connect)
                sigma[indI] -= phase * Kpq * ci_vec[indJ];
        }
    }
}