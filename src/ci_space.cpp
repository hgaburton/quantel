#include "ci_space.h"

void CIspace::initialize(std::string citype)
{
    // Build the FCI space
    if(citype == "FCI") 
        build_fci_determinants();
    else 
        throw std::runtime_error("CI space type not implemented");
    // Build memory maps
    build_memory_map(true);
    build_memory_map(false);
}

void CIspace::build_fci_determinants()
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
}

void CIspace::build_memory_map(bool alpha)
{
    // Populate m_map with connected determinants
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=p; q<m_nmo; q++)
    {
        // Pair keys
        Excitation key1_pq = {p,q,alpha};
        Excitation key1_qp = {q,p,alpha};
        // Initialise map vectors
        m_map1[key1_pq] = std::vector<std::tuple<size_t,size_t,int> >();
        if(p!=q) m_map1[key1_qp] = std::vector<std::tuple<size_t,size_t,int> >();

        // Make an exitation
        Excitation Epq = {p,q,alpha};
        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            // Get alfa excitation
            auto detI = detJ.get_excitation(Epq);
            size_t indI = m_dets[std::get<0>(detI)];
            double phase = std::get<1>(detI);
            if(phase != 0) 
            {
                m_map1[key1_pq].push_back(std::make_tuple(indJ,indI,phase));
                if(p!=q) m_map1[key1_qp].push_back(std::make_tuple(indI,indJ,phase));
            }
        } 
    }

/*
    // Populate m_map with connected determinants
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Pair keys
        Excitation key1_alfa = {p,q,true};
        Excitation key1_beta = {p,q,false};
        // Initialise map vectors
        m_map1[key1_alfa] = std::vector<std::tuple<size_t,size_t,int> >();
        m_map1[key1_beta] = std::vector<std::tuple<size_t,size_t,int> >();
        // Make an exitation
        Excitation Epq_alfa = {p,q,true};
        Excitation Epq_beta = {p,q,false};
        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            // Get alfa excitation
            auto det_a = detJ.get_excitation(Epq_alfa);
            if(std::get<1>(det_a) != 0) 
                m_map1[key1_alfa].push_back(
                    std::make_tuple(indJ,m_dets[std::get<0>(det_a)],std::get<1>(det_a)));
            // Get beta excitation
            auto det_b = detJ.get_excitation(Epq_beta);
            if(std::get<1>(det_b) != 0) 
                m_map1[key1_beta].push_back(
                    std::make_tuple(indJ,m_dets[std::get<0>(det_b)],std::get<1>(det_b)));
        } 
    }

        // Loop over second pair of indices
        for(size_t r=0; r<m_nmo; r++)
        for(size_t s=0; s<m_nmo; s++)
        {
            // Pair keys
            Excitation key2_alfa = {r,s,true};
            Excitation key2_beta = {r,s,false};
            auto key_aa = std::make_tuple(key1_alfa,key2_alfa);
            auto key_ab = std::make_tuple(key1_alfa,key2_beta);
            auto key_bb = std::make_tuple(key1_beta,key2_beta);
            // Initialise map vectors
            m_map2[key_aa] = std::vector<std::tuple<size_t,size_t,int> >();
            m_map2[key_ab] = std::vector<std::tuple<size_t,size_t,int> >();
            m_map2[key_bb] = std::vector<std::tuple<size_t,size_t,int> >();
            // Make second exitation
            Excitation Ers_alfa = {r,s,true};
            Excitation Ers_beta = {r,s,false};
            // Loop over determinants
            for(auto &[detJ, indJ] : m_dets)
            {
                // Get alfa-alfa excitation
                auto det_aa = detJ.get_multiple_excitations({Ers_alfa,Epq_alfa});
                if(std::get<1>(det_aa) != 0) 
                    m_map2[key_aa].push_back(
                        std::make_tuple(indJ,m_dets[std::get<0>(det_aa)],std::get<1>(det_aa)));
                // Get alfa-beta excitation
                auto det_ab = detJ.get_multiple_excitations({Ers_beta,Epq_alfa});
                if(std::get<1>(det_ab) != 0) 
                    m_map2[key_ab].push_back(
                        std::make_tuple(indJ,m_dets[std::get<0>(det_ab)],std::get<1>(det_ab)));
                // Get beta-beta excitation
                auto det_bb = detJ.get_multiple_excitations({Ers_beta,Epq_beta});
                if(std::get<1>(det_bb) != 0) 
                    m_map2[key_bb].push_back(
                        std::make_tuple(indJ,m_dets[std::get<0>(det_bb)],std::get<1>(det_bb)));
            }   
        }
    }
*/
}