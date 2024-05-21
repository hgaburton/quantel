#include <omp.h>
#include "ci_expansion.h"
#include "fmt/format.h"

void CIexpansion::define_space(std::vector<Determinant> det_list)
{ 
    // Set the number of determinants
    m_ndet = det_list.size();

    // Initialise determinants in map
    for(size_t index = 0; index < m_ndet; index++)
    {
        // Get reference to determinant
        Determinant &det = det_list[index];
        // Check number of orbitals
        if(det.nmo() != m_nmo)
            throw std::runtime_error("Determinant has wrong number of orbitals");
        // Add determinant to maps
        m_dets[det] = 0;
    }

    // Now setup indexing for main list
    size_t index = 0;
    for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        it->second = index++;

    // Setup auxiliary determinants
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Get excitation
        std::tuple<int,int> Epq(p,q);
        // Loop over determinants
        for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        {
            // Get determinant
            const Determinant &detI = it->first;

            // Get alfa excitation
            auto Epq_det_alfa = detI.get_excitation(Epq,true);
            // Add to auxiliary list
            if(std::get<1>(Epq_det_alfa) != 0)
                m_aux_dets[std::get<0>(Epq_det_alfa)] = 0;

            // Get beta excitation
            auto Epq_det_beta = detI.get_excitation(Epq,false);
            // Add to auxiliary list
            if(std::get<1>(Epq_det_beta) != 0)
                m_aux_dets[std::get<0>(Epq_det_beta)] = 0;
        }
    }

    // Get number of auxiliary determinants
    m_det_aux = m_aux_dets.size();

    // Setup indexing for auxiliary list
    index = 0;
    for(auto it = m_aux_dets.begin(); it != m_aux_dets.end(); it++)
        it->second = index++;

}

void CIexpansion::print()
{
    std::cout << "Main determinant list" << std::endl;
    for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        std::cout << it->first.str() << std::endl;

    std::cout << "Auxiliary determinant list" << std::endl;
    for(auto it = m_aux_dets.begin(); it != m_aux_dets.end(); it++)
        std::cout << it->first.str() << std::endl;
}

void CIexpansion::print_vector(std::vector<double> &ci_vec)
{
    // Check size of input
    assert(ci_vec.size() == m_ndet);
    // Print vector
    for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        fmt::print("{:>s}: {:>10.6f}\n", it->first.str(), ci_vec[it->second]);
}

void CIexpansion::sigma_vector(std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    // Check size of input
    assert(ci_vec.size() == m_ndet);
    // Resize output
    sigma.resize(m_ndet);
    // Get one-electron part of sigma vector
    sigma_one_electron(ci_vec, sigma);
    sigma_two_electron(ci_vec, sigma);
}

void CIexpansion::sigma_one_electron(std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    std::cout << "One-electron part of sigma vector" << std::endl;

    // Outer loop over operators
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Get excitation and coefficient
        std::tuple<int,int> Epq(p,q);
        double hpq_a = m_ints.oei(p,q,true);
        double hpq_b = m_ints.oei(p,q,false);

        // Inner loop over determinants
        #pragma omp parallel for
        for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        {
            // Get determinant and index
            const Determinant &detI = it->first;
            const size_t indexI = it->second;

            // Get alfa excitation, determinant, and phase
            {
                auto Epq_det = detI.get_excitation(Epq,true);
                const Determinant &detJ = std::get<0>(Epq_det);
                const int phase = std::get<1>(Epq_det);
                // Add contribution if excitation is allowed
                if((phase != 0) and (m_dets.find(detJ) != m_dets.end()))
                {
                    const size_t indexJ = m_dets[detJ];
                    sigma[indexJ] += phase * hpq_a * ci_vec[indexI];
                }
            }

            // Get beta excitation, determinant, and phase
            {
                auto Epq_det = detI.get_excitation(Epq,false);
                const Determinant &detJ = std::get<0>(Epq_det);
                const int phase = std::get<1>(Epq_det);
                // Add contribution if excitation is allowed
                if((phase != 0) and (m_dets.find(detJ) != m_dets.end()))
                {
                    const size_t indexJ = m_dets[detJ];
                    sigma[indexJ] += phase * hpq_b * ci_vec[indexI];
                }
            }
        }
    }
}

void CIexpansion::sigma_two_electron(std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    std::cout << "Two-electron part of sigma vector" << std::endl;
    // Outer loop over operators
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Get excitations and coefficient
        std::tuple<int,int> Epr(p,r);
        std::tuple<int,int> Eqs(q,s);
        std::tuple<int,int> Eps(p,s);
        std::cout << "[" << r << "->" << p << "] [" << s << "->" << q << "]" << std::endl;

        // Get coefficients
        double vpqrs_aa = m_ints.tei(p,q,r,s,true,true);
        double vpqrs_ab = m_ints.tei(p,q,r,s,true,false);
        double vpqrs_ba = m_ints.tei(p,q,r,s,false,true);
        double vpqrs_bb = m_ints.tei(p,q,r,s,false,false);

        // Inner loop over determinants
        for(auto it = m_dets.begin(); it != m_dets.end(); it++)
        {
            // Get determinant and index
            const Determinant &detI = it->first;
            const size_t indexI = it->second;
            double coeff = ci_vec[indexI];
            std::cout << "Determinant: " << detI.str() << std::endl;

            // Perform first alfa excitation
            auto Eqsa_det = detI.get_excitation(Eqs,true);

            std::cout << "Excitation 1: " << std::get<0>(Eqsa_det).str() << " phase " << std::get<1>(Eqsa_det) << std::endl;
            if(std::get<1>(Eqsa_det) != 0)
            {
                // Perform second alfa excitation
                auto EpraEqsa_det = std::get<0>(Eqsa_det).get_excitation(Epr,true);
                std::cout << "Excitation 2: " << std::get<0>(EpraEqsa_det).str() 
                          << " phase " << std::get<1>(EpraEqsa_det) << std::endl;

                const Determinant &detJaa = std::get<0>(EpraEqsa_det);
                const int phase_aa = std::get<1>(EpraEqsa_det) * std::get<1>(Eqsa_det);
                // Include contribution if determinant exists
                if(m_dets.find(detJaa) != m_dets.end())
                {
                    const size_t indexJaa = m_dets[detJaa];
                    sigma[indexJaa] += 0.25 * phase_aa * vpqrs_aa * coeff;
                }

                // Perform second beta excitation
                auto EprbEqsa_det = std::get<0>(Eqsa_det).get_excitation(Epr,false);
                const Determinant &detJba = std::get<0>(EprbEqsa_det);
                const int phase_ba = std::get<1>(EprbEqsa_det) * std::get<1>(Eqsa_det);
                // Include contribution if determinant exists
                if(m_dets.find(detJba) != m_dets.end())
                {
                    const size_t indexJba = m_dets[detJba];
                    sigma[indexJba] += 0.25 * phase_ba * vpqrs_ba * coeff;
                }
            }

            // Perform first beta excitation
            auto Eqsb_det = detI.get_excitation(Eqs,false);
            if(std::get<1>(Eqsb_det) != 0)
            {
                // Perform second alfa excitation
                auto EpraEqsb_det = std::get<0>(Eqsb_det).get_excitation(Epr,true);
                const Determinant &detJab = std::get<0>(EpraEqsb_det);
                const int phase_ab = std::get<1>(EpraEqsb_det) * std::get<1>(Eqsb_det);
                // Include contribution if determinant exists
                if(m_dets.find(detJab) != m_dets.end())
                {
                    const size_t indexJab = m_dets[detJab];
                    sigma[indexJab] += 0.25 * phase_ab * vpqrs_ab * coeff;
                }

                // Perform second beta excitation
                auto EprbEqsb_det = std::get<0>(Eqsb_det).get_excitation(Epr,false);
                const Determinant &detJbb = std::get<0>(EprbEqsb_det);
                const int phase_bb = std::get<1>(EprbEqsb_det) * std::get<1>(Eqsb_det);
                // Include contribution if determinant exists
                if(m_dets.find(detJbb) != m_dets.end())
                {
                    const size_t indexJbb = m_dets[detJbb];
                    sigma[indexJbb] += 0.25 * phase_bb * vpqrs_bb * coeff;
                }
            }

            // Exchange contribution
            if(q == r)
            {
                // Get alfa excitation, determinant, and phase
                {
                    auto Eps_det = detI.get_excitation(Eps,true);
                    const Determinant &detJ = std::get<0>(Eps_det);
                    const int phase = std::get<1>(Eps_det);
                    // Add contribution if excitation is allowed
                    if((phase != 0) and (m_dets.find(detJ) != m_dets.end()))
                    {
                        const size_t indexJ = m_dets[detJ];
                        sigma[indexJ] -= 0.25 * phase * vpqrs_aa * coeff;
                    }
                }

                // Get beta excitation, determinant, and phase
                {
                    auto Eps_det = detI.get_excitation(Eps,false);
                    const Determinant &detJ = std::get<0>(Eps_det);
                    const int phase = std::get<1>(Eps_det);
                    // Add contribution if excitation is allowed
                    if((phase != 0) and (m_dets.find(detJ) != m_dets.end()))
                    {
                        const size_t indexJ = m_dets[detJ];
                        sigma[indexJ] -= 0.25 * phase * vpqrs_bb * coeff;
                    }
                }

            }
        }
    }
}