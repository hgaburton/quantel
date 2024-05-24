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
    auto &m_map = get_map(alpha);

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
            {
                // If the detI is in the CI space, add connection to the map
                try {
                    size_t indI = m_dets.at(detI);
                    m_map[Epq].push_back(std::make_tuple(indJ,indI,phase));
                } catch(const std::out_of_range& e) { }
            }
        } 
    }
}

void CIspace::build_memory_map2(bool alpha1, bool alpha2)
{
    // Get relevant memory map
    assert(alpha1 >= alpha2);
    auto &m_map = get_map(alpha1,alpha2);

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

        // Initialise map vectors
        #pragma omp critical 
        {
            m_map[Epqrs] = std::vector<std::tuple<size_t,size_t,int> >();
        }

        // Loop over determinants
        for(auto &[detJ, indJ] : m_dets)
        {
            Determinant detI = detJ;
            int phase = detI.apply_excitation(Epqrs,alpha1,alpha2);
            if(phase != 0) 
            {
                // If the detI is in the CI space, add connection to the map
                try {
                    size_t indI = m_dets.at(detI);
                    m_map[Epqrs].push_back(std::make_tuple(indJ,indI,phase));                      
                } catch(const std::out_of_range& e) { }
            }
        }
    }
}

/// Print the determinant list
void CIspace::print() const 
{
    for(auto &[det, index] : m_dets)
        std::cout << det_str(det) << ": " << index << std::endl;
}

/// Print a CI vector
void CIspace::print_vector(const std::vector<double> &ci_vec, double tol) const
{
    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace::print_vector() CI vector size error");
    
    for(auto &[det, ind] : m_dets)
    {
        if(std::abs(ci_vec[ind]) > tol) 
            fmt::print("{:>s}: {:>10.6f}\n", det_str(det), ci_vec[ind]);;
    }   
}


void CIspace::H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    // Check size of input
    if(ci_vec.size() != m_ndet) 
        throw std::runtime_error("CIspace::H_on_vec() CI vector size error");

    // Resize output
    sigma.resize(m_ndet,0.0);

    // Compute scalar part of sigma vector
    H0_on_vec(ci_vec, sigma);
    // Get one-electron part of sigma vector
    H1_on_vec(ci_vec, sigma, true);
    H1_on_vec(ci_vec, sigma, false);
    // Get one-electron part of sigma vector
    H2_on_vec(ci_vec, sigma, true, false);
    H2_on_vec(ci_vec, sigma, true, true);
    H2_on_vec(ci_vec, sigma, false, false);
}

void CIspace::H0_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma)
{
    double v_scalar = m_ints.scalar_potential();
    for(size_t ind=0; ind<m_ndet; ind++)
        sigma[ind] += ci_vec[ind] * v_scalar;
}

void CIspace::H1_on_vec(
    const std::vector<double> &ci_vec, std::vector<double> &sigma, bool alpha)
{
    // Get relevant memory map
    auto &m_map = get_map(alpha);
    double tol = m_ints.tol();

    // Get one-electron integrals
    for(size_t p=0; p<m_nmo; p++)
    for(size_t q=0; q<m_nmo; q++)
    {
        // Get one-electron integral
        double hpq = m_ints.oei(p,q,alpha);
        if(std::abs(hpq) < tol) continue;
        // Loop over connections
        for(auto &[indJ, indI, phase] : m_map.at({p,q}))
            sigma[indI] += phase * hpq * ci_vec[indJ];
    }
}

void CIspace::H2_on_vec(
    const std::vector<double> &ci_vec, std::vector<double> &sigma, 
    bool alpha1, bool alpha2)
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
        
        // Get two-electron integral
        double vpqrs = scale * m_ints.tei(p,q,r,s,alpha1,alpha2);
        double vrspq = scale * m_ints.tei(r,s,p,q,alpha1,alpha2);

        if((std::abs(vpqrs) < tol) and (std::abs(vrspq) < tol)) 
            continue;

        // Loop over connections
        for(auto &[indJ, indI, phase] : m_map.at({p,q,r,s}))
        {
            sigma[indI] += phase * vpqrs * ci_vec[indJ];
            if(pq!=rs) sigma[indJ] += phase * vrspq * ci_vec[indI];
        }
    }
}

void CIspace::build_Hmat(std::vector<double> &Hmat)
{
    // Check size of output and initialise memory
    Hmat.resize(m_ndet*m_ndet,0.0);

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
    #pragma omp parallel for
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
        double hpq = m_ints.oei(p,q,alpha);
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

        double vpqrs = scale * m_ints.tei(p,q,r,s,alpha1,alpha2);
        double vrspq = scale * m_ints.tei(r,s,p,q,alpha1,alpha2);

        if((std::abs(vpqrs) < tol) and (std::abs(vrspq) < tol)) 
            continue;

        for(auto &[indJ, indI, phase] : m_map.at({p,q,r,s}))
        {
            H2[indI*m_ndet+indJ] += phase * vpqrs;
            if(pq!=rs) H2[indJ*m_ndet+indI] += phase * vrspq;
        }
    }
}

void CIspace::build_rdm1(
    const std::vector<double> &bra, const std::vector<double> &ket,
    std::vector<double> &rdm1, bool alpha)
{
    // Check size of input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace::build_rdm1() bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace::build_rdm1() ket vector size error");

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
    // Check input
    if(bra.size() != m_ndet) 
        throw std::runtime_error("CIspace::build_rdm2() bra vector size error");
    if(ket.size() != m_ndet)
        throw std::runtime_error("CIspace::build_rdm2() ket vector size error");
    if(alpha1 < alpha2) 
        throw std::runtime_error("CIspace::build_rdm2() Cannot compute rdm2_ba, try rdm2_ab instead");

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