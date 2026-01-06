#include "ci_space.h"
#include "omp_device.h"
#include <cstdint>
#include <omp.h>

void CIspace::initialize(std::string citype, std::vector<std::string> detlist)
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

    // Build memory maps
    build_memory_map1();
    build_memory_map2();
   
    // Record that we successfully initialised
    m_initialized = true;
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

void CIspace::build_cis_determinants(bool with_ref)
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

void CIspace::build_custom_determinants(std::vector<std::string> detlist)
{
    m_ndet = detlist.size();
    for(size_t idet=0; idet<m_ndet; idet++)
        m_dets[Determinant(detlist[idet])] = idet;
}


void CIspace::build_memory_map1()
{
    // Populate m_map with connected determinants
    //#pragma omp parallel for collapse(2)
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
            m_map_a[Epq] = std::vector<std::tuple<size_t,size_t,int> >();
            m_map_b[Epq] = std::vector<std::tuple<size_t,size_t,int> >();
            if(Epq != Eqp)
            {
                m_map_a[Eqp] = std::vector<std::tuple<size_t,size_t,int> >();
                m_map_b[Eqp] = std::vector<std::tuple<size_t,size_t,int> >();
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
                        m_map_a[Epq].push_back(std::make_tuple(indJ,indI,phase));
                        if(Epq != Eqp)
                            m_map_a[Eqp].push_back(std::make_tuple(indI,indJ,phase));
                    } catch(const std::out_of_range& e) { }
            }
            {   // Beta
                Determinant detI = detJ;
                // Get alfa excitation
                int phase = detI.apply_excitation(Epq,false);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_map_b[Epq].push_back(std::make_tuple(indJ,indI,phase));
                        if(Epq != Eqp)
                            m_map_b[Eqp].push_back(std::make_tuple(indI,indJ,phase));
                    } catch(const std::out_of_range& e) { }
            }

        } 
    }
}

void CIspace::build_memory_map2()
{
    // Populate m_map with connected determinants
    //#pragma omp parallel for collapse(4)
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

        // Initialise map vectors
        #pragma omp critical 
        {
            m_map_aa[Epqrs] = std::vector<std::tuple<size_t,size_t,int> >();
            m_map_ab[Epqrs] = std::vector<std::tuple<size_t,size_t,int> >();
            m_map_bb[Epqrs] = std::vector<std::tuple<size_t,size_t,int> >();
        }

        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            {   // Alfa-Alfa
                Determinant detI = detJ;
                int phase = detI.apply_excitation(Epqrs,true,true);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_map_aa[Epqrs].push_back(std::make_tuple(indJ,indI,phase));                      
                    } catch(const std::out_of_range& e) { }
            }
            {   // Alfa-Beta
                Determinant detI = detJ;
                int phase = detI.apply_excitation(Epqrs,true,false);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_map_ab[Epqrs].push_back(std::make_tuple(indJ,indI,phase));                      
                    } catch(const std::out_of_range& e) { }
            }
            {   // Beta-Beta
                Determinant detI = detJ;
                int phase = detI.apply_excitation(Epqrs,false,false);
                if(phase != 0) 
                    try {
                        size_t indI = m_dets.at(detI);
                        m_map_bb[Epqrs].push_back(std::make_tuple(indJ,indI,phase));                      
                    } catch(const std::out_of_range& e) { }
            }
        }
    }
}

/// Print the determinant list
void CIspace::print() const 
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    for(auto &[det, index] : m_dets)
        std::cout << det_str(det) << ": " << index << std::endl;
}

/// Print a CI vector
void CIspace::print_vector(const std::vector<double> &ci_vec, double tol) const
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace::print_vector: CI vector size error");
    
    for(auto &[det, ind] : m_dets)
    {
        if(std::abs(ci_vec[ind]) > tol) 
            fmt::print("{:>s}: {:>10.6f}\n", det_str(det), ci_vec[ind]);;
    }   
}


void CIspace::H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of input
    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace::H_on_vec: CI vector size error");

    // Get information about the OpenMP device
    omp_device dev;
    // Tolerance 
    double tol = m_ints.tol();
    
    // Get thread-safe memory for sigma vector
    std::vector<double> sigma_t(dev.nthreads*m_ndet);
    std::fill(sigma_t.begin(), sigma_t.end(), 0.0);
    
    // One-electron part
    //#pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Access memory 
        size_t ithread = dev.thread_id();
        double *st = &sigma_t[ithread*m_ndet];

        double hpq = m_ints.oei(p,q);
        if(std::abs(hpq) > tol)
        {
            // Alfa contribution
            for(auto &[indJ, indI, phase] : m_map_a.at({p,q}))
                st[indI] += phase * hpq * ci_vec[indJ];
            // Beta contribution
            for(auto &[indJ, indI, phase] : m_map_b.at({p,q}))
                st[indI] += phase * hpq * ci_vec[indJ];
        }
    }

    // Two-electron part
    //#pragma omp parallel for collapse(4)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*m_nmo + q;
        size_t rs = r*m_nmo + s;
        if(pq > rs) continue;

        // Access memory 
        size_t ithread = dev.thread_id();
        double *st = &sigma_t[ithread*m_ndet];
        
        // Get two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>
        // Note the different prefactors for same-spin and opposite-spin terms
        double vpqrs_same = 0.25 * (m_ints.tei(p,q,r,s) - m_ints.tei(p,q,s,r));
        double vrspq_same = 0.25 * (m_ints.tei(r,s,p,q) - m_ints.tei(s,r,p,q));
        double vpqrs_diff = 1.0 * m_ints.tei(p,q,r,s);
        double vrspq_diff = 1.0 * m_ints.tei(r,s,p,q);
        // Alfa-alfa
        if(true) //(std::abs(vpqrs_same) > tol) or (std::abs(vrspq_same) > tol))
            for(auto &[indJ, indI, phase] : m_map_aa.at({p,q,r,s}))
            {
                st[indI] += phase * vpqrs_same * ci_vec[indJ];
                if(pq!=rs) st[indJ] += phase * vrspq_same * ci_vec[indI];
            }
        
        // alfa-beta
        if(true) //(std::abs(vpqrs_diff) > tol) and (std::abs(vrspq_diff) > tol))
            for(auto &[indJ, indI, phase] : m_map_ab.at({p,q,r,s}))
            {
                st[indI] += phase * vpqrs_diff * ci_vec[indJ];
                if(pq!=rs) st[indJ] += phase * vrspq_diff * ci_vec[indI];
            }

        // beta-beta
        if(true) //(std::abs(vpqrs_same) > tol) and (std::abs(vrspq_same) > tol))
            for(auto &[indJ, indI, phase] : m_map_bb.at({p,q,r,s}))
            {
                st[indI] += phase * vpqrs_same * ci_vec[indJ];
                if(pq!=rs) st[indJ] += phase * vrspq_same * ci_vec[indI];
            }
    }

    // Initialise resulting sigma vector
    sigma.resize(m_ndet,0.0);
    double v_scalar = m_ints.scalar_potential();
    for(size_t ind=0; ind<m_ndet; ind++)
        sigma[ind] += ci_vec[ind] * v_scalar;

    // Compile results from all threads
    for(size_t ind=0; ind<m_ndet; ind++)
    for(size_t ithread=0; ithread<dev.nthreads; ithread++)
        sigma[ind] += sigma_t[ithread*m_ndet+ind];
}

void CIspace::build_Hmat(std::vector<double> &Hmat)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of output and initialise memory
    Hmat.resize(m_ndet*m_ndet);
    std::fill(Hmat.begin(), Hmat.end(), 0.0);

    // Scalar component
    build_H0(Hmat);
    // One-electron component
    build_H1(Hmat,true);
    build_H1(Hmat,false);
    // Two-electron component
    build_H2(Hmat,true,false);
    build_H2(Hmat,true,true);
    build_H2(Hmat,false,false);
} 

void CIspace::build_H0(std::vector<double> &H0)
{
    // Diagonal scalar part
    double v_scalar = m_ints.scalar_potential();
    //#pragma omp parallel for
    for(size_t I=0; I<m_ndet; I++)
        H0[I*m_ndet+I] += v_scalar;
}

void CIspace::build_H1(std::vector<double> &H1, bool alpha)
{
    // Get relevant memory map
    auto &m_map = get_map(alpha);
    double tol = m_ints.tol();

    // Get one-electron integrals
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=p; q<m_nmo; q++)
    {
        double hpq = m_ints.oei(p,q);
        if(std::abs(hpq) < tol) continue;

        for(auto &[indJ, indI, phase] : m_map.at({p,q}))
        {
            H1[indI*m_ndet+indJ] += phase * hpq;
            if(p!=q) 
                H1[indJ*m_ndet+indI] += phase * hpq;
        }
    }
}

void CIspace::build_H2(std::vector<double> &H2, bool alpha1, bool alpha2)
{
    assert(alpha1 >= alpha2);
    double tol = m_ints.tol();

    // Get relevant memory map
    auto &m_map = get_map(alpha1,alpha2);
    
    // Scaling factor for same spin terms is 1/4 due to antisymmetrisation
    double scale = (alpha1 == alpha2) ? 0.25 : 1.0;

    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*m_nmo + q;
        size_t rs = r*m_nmo + s;
        if(pq > rs) continue;

        double vpqrs = (alpha1==alpha2) ? (m_ints.tei(p,q,r,s)-m_ints.tei(p,q,s,r)) : m_ints.tei(p,q,r,s);
        double vrspq = (alpha1==alpha2) ? (m_ints.tei(r,s,p,q)-m_ints.tei(r,s,q,p)) : m_ints.tei(r,s,p,q);

        if((std::abs(vpqrs) < tol) and (std::abs(vrspq) < tol)) 
            continue;

        for(auto &[indJ, indI, phase] : m_map.at({p,q,r,s}))
        {
            H2[indI*m_ndet+indJ] += scale * phase * vpqrs;
            if(pq!=rs) H2[indJ*m_ndet+indI] += scale * phase * vrspq;
        }
    }
}

void CIspace::build_rdm1(
    const std::vector<double> &bra, const std::vector<double> &ket,
    std::vector<double> &rdm1, bool alpha)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check size of input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace::build_rdm1: bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace::build_rdm1: ket vector size error");

    // Resize output
    rdm1.resize(m_nmo*m_nmo,0.0);

    // Get relevant memory map
    auto &m_map = get_map(alpha);

    // Compute 1RDM
    #pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        double &rdm_pq = rdm1[p*m_nmo+q];
        for(auto &[indJ, indI, phase] : m_map.at({p,q}))
            rdm_pq += phase * bra[indI] * ket[indJ];
    }
}

void CIspace::build_rdm2(
    const std::vector<double> &bra, const std::vector<double> &ket,
    std::vector<double> &rdm2, bool alpha1, bool alpha2)
{
    if(!m_initialized)
        throw std::runtime_error("CI space has not been initialized!");

    // Check input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace::build_rdm2: bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace::build_rdm2: ket vector size error");
    if(alpha1 < alpha2) 
        throw std::runtime_error("CIspace::build_rdm2: Cannot compute rdm2_ba, try rdm2_ab instead");

    // Resize output
    rdm2.resize(m_nmo*m_nmo*m_nmo*m_nmo,0.0);

    // Get relevant memory map
    auto &m_map = get_map(alpha1,alpha2);

    // Compute 2RDM
    #pragma omp parallel for collapse(4)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t r=0; r<m_nmo; r++)
    for(size_t q=0; q<m_nmo; q++)
    for(size_t s=0; s<m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*m_nmo + q;
        size_t rs = r*m_nmo + s;
        if(pq > rs) continue;

        double &rdm_pqrs = rdm2[p*m_nmo*m_nmo*m_nmo+q*m_nmo*m_nmo+r*m_nmo+s];
        double &rdm_rspq = rdm2[r*m_nmo*m_nmo*m_nmo+s*m_nmo*m_nmo+p*m_nmo+q];

        for(auto &[indJ, indI, phase] : m_map.at({p,q,r,s}))
        {
            rdm_pqrs += phase * bra[indI] * ket[indJ];
            if(pq!=rs) rdm_rspq += phase * bra[indJ] * ket[indI];
        }
    }
}

void CIspace::build_Hd(std::vector<double> &Hdiag)
{
    // Get information about the OpenMP device
    omp_device dev;
    
    // Get thread-safe memory
    std::vector<double> Hdiag_t(dev.nthreads*m_ndet);
    std::fill(Hdiag_t.begin(), Hdiag_t.end(), 0.0);

    // Add one-electron part (only diagonals contribute)
    double tol = m_ints.tol();
    #pragma omp parallel for
    for(size_t p=0; p<m_nmo; p++)
    {
        // Get thread memory buffer
        int ithread = dev.thread_id();
        double *Ht = &Hdiag_t[ithread*m_ndet];

        // Get one-electron alfa integral
        double hpp = m_ints.oei(p,p);
        // Alfa contribution
        for(auto &[indJ, indI, phase] : m_map_a.at({p,p}))
            Ht[indI] += phase * hpp;
        // Beta contribution
        for(auto &[indJ, indI, phase] : m_map_b.at({p,p}))
            Ht[indI] += phase * hpp;
    }

    #pragma omp parallel for collapse(2)
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
       // Get thread memory buffer
       int ithread = dev.thread_id();
       double *Ht = &Hdiag_t[ithread*m_ndet];

        // only include number-preserving terms
        double vpqpq_same = 0.5 * (m_ints.tei(p,q,p,q) - m_ints.tei(p,q,q,p));
        double vpqpq_diff = m_ints.tei(p,q,p,q);
        // Same spin
        if((std::abs(vpqpq_same) > tol))
        {
            // Alfa-Alfa
            for(auto &[indJ, indI, phase] : m_map_aa.at({p,q,p,q}))
                Ht[indI] += phase * vpqpq_same;
            // Beta-Beta
            for(auto &[indJ, indI, phase] : m_map_bb.at({p,q,p,q}))
                Ht[indI] += phase * vpqpq_same;
        }
        // Alfa-Beta
        if(std::abs(vpqpq_diff) > tol) 
            for(auto &[indJ, indI, phase] : m_map_ab.at({p,q,p,q}))
                Ht[indI] += phase * vpqpq_diff;
    }

    // Collect final results
    Hdiag.resize(m_ndet);
    std::fill(Hdiag.begin(), Hdiag.end(), m_ints.scalar_potential());
    for(size_t it=0; it < dev.nthreads; it++)
    for(size_t ind=0; ind < m_ndet; ind++)
        Hdiag[ind] += Hdiag_t[it*m_ndet+ind];
}