#include "ci_space2.h"
#include "omp_device.h"
#include <cstdint>
#include <omp.h>

void CIspace2::initialize(std::string citype, std::vector<std::string> detlist)
{
    // Check that we haven't already initialized CI space
    if(m_initialized)
        throw std::runtime_error("CI space has already been initialised!");

    // Transform CI type to uppercase
    std::transform(citype.begin(), citype.end(), citype.begin(),
                   [](unsigned char c){ return std::toupper(c); });
                   
    // Reject attempts to build non-custom determinant set with custom list
    if((citype!="CUSTOM" and citype!="MRCISD") and detlist.size()>0)
        throw std::runtime_error("Custom determinant list not compatible with requested CI type");
    // Build the FCI space
    if(citype == "FCI") 
        build_fci_determinants();
    else if(citype == "ESMF")
        // Include reference determinant in CIS 
        build_cis_determinants(true);
    else if(citype == "CIS")
        // Exclude reference determinant in CIS
        build_cis_determinants(false);
    else if(citype == "CUSTOM")
        // Build custom determinant list 
        build_custom_determinants(detlist);
    else 
        throw std::runtime_error("CI space type not implemented");

    // Build auxiliary determinants
    build_auxiliary_determinants();

    // Build memory maps
    build_memory_map();
   
    // Record that we successfully initialised
    m_initialized = true;
}

void CIspace2::build_fci_determinants()
{ 
    // Populate m_det with FCI space
    m_ndet = 0;
    std::vector<uint8_t> occ_alfa(m_nmo,false);
    std::vector<uint8_t> occ_beta(m_nmo,false);
    std::fill_n(occ_alfa.begin(), m_nalfa, 1);
    std::fill_n(occ_beta.begin(), m_nbeta, 1);
    do {
        do {
            m_dets[Determinant(occ_alfa,occ_beta)] = m_ndet++;
        } while(std::prev_permutation(occ_beta.begin(), occ_beta.end()));
    } while(std::prev_permutation(occ_alfa.begin(), occ_alfa.end()));
}

void CIspace2::build_cis_determinants(bool with_ref)
{
    // Populate m_det with CIS space
    m_ndet = 0;
    
    // Setup reference occupation vectors
    std::vector<uint8_t> ref_alfa(m_nmo,false);
    std::vector<uint8_t> ref_beta(m_nmo,false);
    std::fill_n(ref_alfa.begin(), m_nalfa, 1);
    std::fill_n(ref_beta.begin(), m_nbeta, 1);

    // Include reference determinant if requested
    if(with_ref) 
        m_dets[Determinant(ref_alfa,ref_beta)] = m_ndet++;

    // Setup alfa excitations
    for(size_t i=0; i<m_nalfa; i++)
    for(size_t a=m_nalfa; a<m_nmo; a++)
    {
        std::vector<uint8_t> occ_alfa = ref_alfa;
        occ_alfa[i] = 0;
        occ_alfa[a] = 1;
        m_dets[Determinant(occ_alfa,ref_beta)] = m_ndet++;
    }

    // Setup beta excitations
    for(size_t i=0; i<m_nbeta; i++)
    for(size_t a=m_nbeta; a<m_nmo; a++)
    {
        std::vector<uint8_t> occ_beta = ref_beta;
        occ_beta[i] = 0;
        occ_beta[a] = 1;
        m_dets[Determinant(ref_alfa,occ_beta)] = m_ndet++;
    }
}

void CIspace2::build_custom_determinants(std::vector<std::string> detlist)
{
    m_ndet = detlist.size();
    for(size_t idet=0; idet<m_ndet; idet++)
        m_dets[Determinant(detlist[idet])] = idet;
}

void CIspace2::build_auxiliary_determinants()
{
    // Append auxiliary determinants to end of m_dets
    m_ndet_aux = m_ndet;
    // Loop over excitations
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        Eph Epq = {p,q};
        // Loop over determinants in CI space
        for(auto &[detJ, indJ] : m_dets)
        {
            {   // Alfa
                Determinant detI = detJ;
                // Get alfa excitation
                int phase = detI.apply_excitation(Epq,true);
                if(phase != 0) 
                    try {
                        // Check if determinant is already in m_dets
                        size_t indI = m_dets.at(detI);
                    } catch(const std::out_of_range& e) {
                        // If not, add it to m_dets
                        m_dets[detI] = m_ndet_aux++;
                    }
            } 
            {   // Beta
                Determinant detI = detJ;
                // Get beta excitation
                int phase = detI.apply_excitation(Epq,false);
                if(phase != 0) 
                    try {
                        // Check if determinant is already in m_dets
                        size_t indI = m_dets.at(detI);
                    } catch(const std::out_of_range& e) {
                        // If not, add it to m_dets
                        m_dets[detI] = m_ndet_aux++;
                    }
            }
        }
    }
}

void CIspace2::build_memory_map()
{
    // Initialise m_det_ex
    m_det_ex.resize(m_ndet_aux);

    // Populate m_map with connected determinants
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        if(q>p) continue;
        // Make an exitation
        Eph Epq = {p,q};
        Eph Eqp = {q,p};

        // Initialise map vectors
        #pragma omp critical 
        {
            m_ex_det[{Epq,true}] = std::vector<std::tuple<size_t,size_t,int> >();
            m_ex_det[{Epq,false}] = std::vector<std::tuple<size_t,size_t,int> >();
            if(Epq != Eqp)
            {
                m_ex_det[{Eqp,true}] = std::vector<std::tuple<size_t,size_t,int> >();
                m_ex_det[{Eqp,false}] = std::vector<std::tuple<size_t,size_t,int> >();
            }
        }

        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            {   // Alfa
                Determinant detI = detJ;
                // Get alfa excitation
                int phase = detI.apply_excitation(Epq,true);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_ex_det[{Epq,true}].push_back(std::make_tuple(indJ,indI,phase));
                        if(Epq != Eqp) 
                            m_ex_det[{Eqp,true}].push_back(std::make_tuple(indI,indJ,phase));
                        #pragma omp critical
                        {
                            m_det_ex[indJ][std::make_tuple(Epq,true)] = std::make_tuple(indI,phase);
                            if(Epq != Eqp) 
                                m_det_ex[indI][std::make_tuple(Eqp,true)] = std::make_tuple(indJ,phase);
                        }    

                    } catch(const std::out_of_range& e) { }
            }
            {   // Beta
                Determinant detI = detJ;
                // Get alfa excitation
                int phase = detI.apply_excitation(Epq,false);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_ex_det[{Epq,false}].push_back(std::make_tuple(indJ,indI,phase));
                        if(Epq != Eqp)
                            m_ex_det[{Eqp,false}].push_back(std::make_tuple(indI,indJ,phase));
                        #pragma omp critical
                        {
                            m_det_ex[indJ][std::make_tuple(Epq,false)] = std::make_tuple(indI,phase);
                            if(Epq != Eqp) 
                                m_det_ex[indI][std::make_tuple(Eqp,false)] = std::make_tuple(indJ,phase);
                        }
                    } catch(const std::out_of_range& e) { }
            }
        } 
    }
    /*
    for(size_t ind=0; ind<m_ndet_aux; ind++)
    {
        std::cout << "Det " << m_det << " excitations:" << std::endl;
        for(auto &[exc, tup] : m_det_ex[ind])
        {
            auto &[p,q] = std::get<0>(exc);
            bool alpha = std::get<1>(exc);
            auto &[ind2, phase] = tup;
            std::cout << "  E[" << p << "," << q << "]_" << (alpha ? "a" : "b") 
                      << " -> Det " << ind2 << " with phase " << phase << std::endl;
        }
    }
    */
}

/// Print the determinant list
void CIspace2::print() const 
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    for(auto &[det, index] : m_dets)
        if(index < m_ndet) std::cout << det_str(det) << ": " << index << std::endl;
}

/// Print the auxiliary determinant list
void CIspace2::print_auxiliary() const
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    for(auto &[det, index] : m_dets)
        std::cout << det_str(det) << ": " << index << std::endl;
}

/// Print a CI vector
void CIspace2::print_vector(const std::vector<double> &ci_vec, double tol) const
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace2::print_vector: CI vector size error");
    
    for(auto &[det, index] : m_dets)
    {
        if(std::abs(ci_vec[index]) > tol and index < m_ndet) 
            fmt::print("{:>s}: {:>10.6f}\n", det_str(det), ci_vec[index]);;
    }   
}

void CIspace2::H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of input
    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace2::H_on_vec: CI vector size error");

    // Get information about the OpenMP device
    omp_device dev;
    // Tolerance 
    double tol = m_ints.tol();
    
    // Get thread-safe memory for sigma vector
    std::vector<double> sigma_t(dev.nthreads*m_ndet_aux);
    std::fill(sigma_t.begin(),sigma_t.end(),0.0);

    // Two-electron part  <pq|rs> <I|Epr|k> <k|Eqs|J> cJ
    #pragma omp parallel for collapse(2)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Access memory 
        size_t ithread = dev.thread_id();
        double *st = &sigma_t[ithread*m_ndet_aux];

        // Access one-electron contribution
        double hqs = m_ints.oei_eff(q,s);

        // Construct auxiliary vector
        std::vector<double> Vqs(m_ndet_aux);
        std::fill(Vqs.begin(),Vqs.end(),0.0);

        // Build auxiliary vector VqsK = <K|Eqs|J> cJ and increment 1-electron term
        for(auto &[indJ, indK, phase] : m_ex_det.at({{q,s},true}))
        {
            if(indJ >= m_ndet) continue;
            Vqs[indK] += phase * ci_vec[indJ];
            st[indK]  += phase * hqs * ci_vec[indJ];
        }
        for(auto &[indK, indJ, phase] : m_ex_det.at({{q,s},false}))
        {
            if(indJ >= m_ndet) continue;
            Vqs[indK] += phase * ci_vec[indJ];
            st[indK]  += phase * hqs * ci_vec[indJ];
        }

        // Loop over p and r to build sigma
        for(size_t p=0; p<m_nmo; p++)
        for(size_t r=0; r<m_nmo; r++)
        {
            // Get value of integral and skip if below threshold
            double vpqrs = m_ints.tei(p,q,r,s);
            if(std::abs(vpqrs) < tol) continue;

            // Apply Epr Eqs
            for(auto &[indK, indI, phase] : m_ex_det.at({{p,r},true}))
                st[indI] += 0.5 * phase * vpqrs * Vqs[indK];
            for(auto &[indK, indI, phase] : m_ex_det.at({{p,r},false}))
                st[indI] += 0.5 * phase * vpqrs * Vqs[indK];
        }
    }

    // Initialise resulting sigma vector
    // Here we extract only first m_ndet elements corresponding to CI space
    // and we ignore the remaining auxiliary determinants
    sigma.resize(m_ndet,0.0);
    double v_scalar = m_ints.scalar_potential();
    for(size_t ind=0; ind<m_ndet; ind++)
        sigma[ind] += ci_vec[ind] * v_scalar;

    // Compile results from all threads
    for(size_t ind=0; ind<m_ndet; ind++)
    for(size_t ithread=0; ithread<dev.nthreads; ithread++)
        sigma[ind] += sigma_t[ithread*m_ndet_aux+ind];
}

void CIspace2::build_Hmat(std::vector<double> &Hmat)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of output and initialise memory
    Hmat.resize(m_ndet*m_ndet);
    std::fill(Hmat.begin(), Hmat.end(), 0.0);

    // Scalar component
    double v_scalar = m_ints.scalar_potential();

    // Loop over bra determinants
    #pragma omp parallel for
    for(size_t indJ=0; indJ<m_ndet; indJ++)
    {
        // Get data buffer
        double *buff = &Hmat[indJ*m_ndet];
        // Scalar term
        buff[indJ] += v_scalar;

        // Loop over connected determinants by Eqs
        for(auto &[exc1, dest] : m_det_ex[indJ])
        {
            auto &[indK, phase1] = dest;
            auto &[Eqs, alpha1] = exc1;
            size_t q = Eqs.particle;
            size_t s = Eqs.hole;

            // Apply single excitation
            if(indK<m_ndet) 
                buff[indK] += m_ints.oei_eff(q,s) * phase1;

            // Loop over connected determinants by Epr
            for(auto &[exc2, dest2] : m_det_ex[indK])
            {
                auto &[indI, phase2] = dest2;
                auto &[Epr, alpha2] = exc2;
                size_t p = Epr.particle;
                size_t r = Epr.hole;

                // Update Hmat
                if(indI < m_ndet)
                    buff[indI] += 0.5 * phase1 * phase2 * m_ints.tei(p,q,r,s);
            }
        }
    }
} 

void CIspace2::build_Hd(std::vector<double> &Hdiag)
{
    // Get information about the OpenMP device
    omp_device dev;
    // Initialise memory
    Hdiag.resize(m_ndet);
    std::fill(Hdiag.begin(), Hdiag.end(), m_ints.scalar_potential());

    // Determinant based
    #pragma omp parallel for
    for(size_t indI=0; indI<m_ndet; indI++)
    {
        // Get thread memory buffer
        double &Ht = Hdiag[indI];

        // Loop over first excitation
        for(auto &[exc1, tup1] : m_det_ex[indI])
        {
            auto &[indJ, phase1] = tup1;
            auto &[Epr, alpha1] = exc1;
            size_t p = Epr.particle;
            size_t r = Epr.hole;

            if(p!=r) continue;
            assert(indJ == indI);

            // One-electron part <I|Epp|I>
            Ht += m_ints.oei(p,p);

            // Loop over second excitation
            for(auto &[exc2, tup2] : m_det_ex[indI])
            {
                auto &[indK, phase2] = tup2;
                auto &[Eqs, alpha2] = exc2;
                size_t q = Eqs.particle;
                size_t s = Eqs.hole;

                if(q!=s) continue;
                assert(indK == indI);

                // Two-electron part <I|Epp Ess|I>
                if(alpha1 == alpha2)
                    Ht += 0.5 * (m_ints.tei(p,q,p,q) - m_ints.tei(p,q,q,p));
                else
                    Ht += 0.5 * m_ints.tei(p,q,p,q);
            }
        }
    }
}

void CIspace2::build_rdm1(
    const std::vector<double> &bra, const std::vector<double> &ket,
    std::vector<double> &rdm1, bool alpha)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace2::build_rdm1: bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace2::build_rdm1: ket vector size error");

    // Resize output
    rdm1.resize(m_nmo*m_nmo,0.0);

    // Compute 1RDM
    #pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        double &rdm_pq = rdm1[p*m_nmo+q];
        for(auto &[indJ, indI, phase] : m_ex_det.at({{p,q},alpha}))
        {
            if(indJ >= m_ndet or indI >= m_ndet) continue;
            rdm_pq += phase * bra[indI] * ket[indJ];
        }
    }
}


void CIspace2::build_rdm2(
    const std::vector<double> &bra, const std::vector<double> &ket,
    std::vector<double> &rdm2, bool alpha1, bool alpha2)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace2::build_rdm2: bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace2::build_rdm2: ket vector size error");
    if(alpha1 < alpha2) 
        throw std::runtime_error("CIspace2::build_rdm2: Cannot compute rdm2_ba, try rdm2_ab instead");

    // thread-safe memory for intermediate quantities
    std::vector<double> rdm2_t(omp_get_max_threads()*m_nmo*m_nmo*m_nmo*m_nmo,0.0);
    std::fill(rdm2_t.begin(), rdm2_t.end(), 0.0);

    // Resize output
    rdm2.resize(m_nmo*m_nmo*m_nmo*m_nmo,0.0);

    // TODO: Block over K index to reduce memory usage
    // Build memory
    std::vector<double> VqsK1(m_nmo*m_nmo*m_ndet_aux,0.0);
    std::vector<double> VqsK2(m_nmo*m_nmo*m_ndet_aux,0.0);
    std::fill(VqsK1.begin(), VqsK1.end(), 0.0);
    std::fill(VqsK2.begin(), VqsK2.end(), 0.0);
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Build VqsK = <K|Eqs|J> cJ
        for(auto &[indJ, indK, phase] : m_ex_det.at({{q,s},alpha2}))
        {
            if(indJ >= m_ndet) continue;
            VqsK1[q*m_nmo*m_ndet_aux+s*m_ndet_aux+indK] += phase * ket[indJ];
        }
        for(auto &[indI, indK, phase] : m_ex_det.at({{q,s},alpha1}))
        {
            if(indI >= m_ndet) continue;
            VqsK2[q*m_nmo*m_ndet_aux+s*m_ndet_aux+indK] += phase * bra[indI];
        }
    }

    // Compute 2RDM
    //#pragma omp parallel for collapse(4)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Get thread
        size_t ithread = omp_get_thread_num();
        double *rdm2_thread = &rdm2_t[ithread*m_nmo*m_nmo*m_nmo*m_nmo];
        for(size_t indK=0; indK<m_ndet_aux; indK++)
        {
            double v1 = VqsK1[q*m_nmo*m_ndet_aux+s*m_ndet_aux+indK];
            double v2 = VqsK2[r*m_nmo*m_ndet_aux+p*m_ndet_aux+indK];
            rdm2_thread[p*m_nmo*m_nmo*m_nmo+q*m_nmo*m_nmo+r*m_nmo+s] += v1 * v2;
        }
    }

    // If alpha1 == alpha2, need to account for exchange term
    if(alpha1 == alpha2)
    {    
        for(size_t p=0; p<m_nmo; p++)
        for(size_t s=0; s<m_nmo; s++)
        {
            // Get thread
            size_t ithread = omp_get_thread_num();
            double *rdm2_thread = &rdm2_t[ithread*m_nmo*m_nmo*m_nmo*m_nmo];
            // Build VqsK = <K|Eqs|J> cJ
            for(auto &[indJ, indI, phase] : m_ex_det.at({{p,s},alpha2}))
            {
                if(indJ >= m_ndet or indI >= m_ndet) continue;
                for(size_t q=0; q<m_nmo; q++)
                    rdm2_thread[p*m_nmo*m_nmo*m_nmo+q*m_nmo*m_nmo+q*m_nmo+s] -= 
                        phase * bra[indI] * ket[indJ];
            }
        }
    }
    
    // Compile results from all threads
    for(size_t pqrs=0; pqrs<m_nmo*m_nmo*m_nmo*m_nmo; pqrs++)
    for(size_t ithread=0; ithread<omp_get_max_threads(); ithread++)
        rdm2[pqrs] += rdm2_t[ithread*m_nmo*m_nmo*m_nmo*m_nmo+pqrs];

}