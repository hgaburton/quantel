#include "ci_space.h"
#include <omp.h>

void CIspace::initialize(std::string citype)
{
    // Build the FCI space
    if(citype == "FCI") 
        build_fci_determinants();
    else 
        throw std::runtime_error("CI space type not implemented");
        
    // Build memory maps
    build_memory_map1(true);
    build_memory_map1(false);
    build_memory_map2(true, true);
    build_memory_map2(true, false);
    build_memory_map2(false, false);
}

void CIspace::build_fci_determinants()
{ 
    // Populate m_det with FCI space
    m_ndet = 0;
    std::vector<uint8_t> occ_alfa(m_nmo,false);
    std::vector<uint8_t> occ_beta(m_nmo,false);
    std::fill_n(occ_alfa.begin(), m_nalfa, 1);
    std::fill_n(occ_beta.begin(), m_nbeta, 1);
    do {
        m_ndeta++;
        do {
            m_ndetb++;
            m_dets[Determinant(occ_alfa,occ_beta)] = m_ndet++;
        } while(std::prev_permutation(occ_beta.begin(), occ_beta.end()));
    } while(std::prev_permutation(occ_alfa.begin(), occ_alfa.end()));
}

void CIspace::build_memory_map1(bool alpha)
{
    // Get relevant memory map
    auto &m_map = alpha ? m_map_a : m_map_b;

    // Populate m_map with connected determinants
    #pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Make an exitation
        Eph Epq = {p,q};
        // Initialise map vectors
        #pragma omp critical 
        {
            m_map[Epq] = std::vector<std::tuple<size_t,size_t,int> >();
        }

        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            // Get copy of determinant
            Determinant detI = detJ;
            // Get alfa excitation
            int phase = detI.apply_excitation(Epq,alpha);
            if(phase != 0) 
                m_map[Epq].push_back(std::make_tuple(indJ,m_dets[detI],phase));
        } 
    }
}

void CIspace::build_memory_map2(bool alpha1, bool alpha2)
{
    // Get relevant memory map
    auto &m_map = alpha1 ? (alpha2 ? m_map_aa : m_map_ab) : m_map_bb;

    // Populate m_map with connected determinants
    #pragma omp parallel for collapse(4)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*m_nmo + q;
        size_t rs = r*m_nmo + s;
        if(pq > rs) continue;

        Epphh Epqrs = {p,q,r,s};
        Epphh Erspq = {r,s,p,q};

        // Initialise map vectors
        #pragma omp critical 
        {
            m_map[Epqrs] = std::vector<std::tuple<size_t,size_t,int> >();
            if(Epqrs!=Erspq)
                m_map[Erspq] = std::vector<std::tuple<size_t,size_t,int> >();
        }

        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            Determinant detI = detJ;
            int phase = detI.apply_excitation(Epqrs,alpha2,alpha1);
            if(phase != 0) 
            {
                m_map[Epqrs].push_back(std::make_tuple(indJ,m_dets[detI],phase));
                if(Epqrs!=Erspq)
                    m_map[Erspq].push_back(std::make_tuple(m_dets[detI],indJ,phase));
            }
        }
    }
}