#include <omp.h>
#include "libint_interface.h"
#include "linalg.h"

using namespace libint2;


void LibintInterface::initialize()
{
    // Save number of basis functions
    m_nbsf = m_basis.nbf();

    // Compute nuclear potential
    compute_nuclear_potential();

    // Compute integrals
    compute_overlap();
    compute_one_electron_matrix();
    compute_two_electron_integrals();

    // Compute the orthogonalisation matrix
    compute_orthogonalization_matrix();
}

double LibintInterface::overlap(size_t p, size_t q) 
{
    return m_S[oei_index(p,q)];
}

double LibintInterface::oei(size_t p, size_t q, bool alpha) 
{
    return alpha ? m_oei_a[oei_index(p,q)] : m_oei_b[oei_index(p,q)];
}

double LibintInterface::tei_J(size_t p, size_t q, size_t r, size_t s) 
{
    return m_tei_J[tei_index(p,q,r,s)];
}

double LibintInterface::tei_K(size_t p, size_t q, size_t r, size_t s) 
{
    return m_tei_K[tei_index(p,q,r,s)];
}


void LibintInterface::set_scalar_potential(double value) 
{
    m_V = value;
}

void LibintInterface::set_ovlp(size_t p, size_t q, double value) 
{ 
    m_S[oei_index(p,q)] = value;
}

void LibintInterface::set_oei(size_t p, size_t q, double value, bool alpha)
{
    std::vector<double> &oei = alpha ? m_oei_a : m_oei_b;
    oei[oei_index(p,q)] = value;
}

void LibintInterface::set_tei_J(size_t p, size_t q, size_t r, size_t s, double value)
{
    m_tei_J[tei_index(p,q,r,s)] = value;
}

void LibintInterface::set_tei_K(size_t p, size_t q, size_t r, size_t s, double value)
{
    m_tei_K[tei_index(p,q,r,s)] = value;
}


void LibintInterface::compute_two_electron_integrals()
{
    // Setup II dimensions
    auto shell2bf = m_basis.shell2bf();
    size_t strides[3];

    // Resize and zero the two-electron integral arrays
    m_tei_J.resize(m_nbsf*m_nbsf*m_nbsf*m_nbsf);
    m_tei_K.resize(m_nbsf*m_nbsf*m_nbsf*m_nbsf);
    std::fill(m_tei_J.begin(), m_tei_J.end(), 0.0);
    std::fill(m_tei_K.begin(), m_tei_K.end(), 0.0);

    // Compute integrals
    Engine coul_engine(Operator::coulomb, m_basis.max_nprim(), m_basis.max_l());
    const auto &buf_coul = coul_engine.results();
    for(size_t s1=0; s1 < m_basis.size(); ++s1)
    for(size_t s2=0; s2 < m_basis.size(); ++s2)
    for(size_t s3=0; s3 < m_basis.size(); ++s3)
    for(size_t s4=0; s4 < m_basis.size(); ++s4)
    {
        // Start of each shell
        size_t bf1 = shell2bf[s1];
        size_t bf2 = shell2bf[s2];
        size_t bf3 = shell2bf[s3];
        size_t bf4 = shell2bf[s4];
        // Size of each shell
        size_t n1 = m_basis[s1].size();
        size_t n2 = m_basis[s2].size();
        size_t n3 = m_basis[s3].size();
        size_t n4 = m_basis[s4].size();
        // Set strides
        strides[0] = n2 * n3 * n4;
        strides[1] = n3 * n4;
        strides[2] = n4;

        // Compute contribution in chemists notation
        coul_engine.compute(m_basis[s1], m_basis[s2], m_basis[s3], m_basis[s4]);
        const auto *ints = buf_coul[0];
        if(ints == nullptr)
            continue;

        // Save elements to array
        for(size_t f1=0; f1 < n1; ++f1)
        for(size_t f2=0; f2 < n2; ++f2)
        for(size_t f3=0; f3 < n3; ++f3)
        for(size_t f4=0; f4 < n4; ++f4)
        {
            // Save value for aabb block in physicists notation
            double value = ints[f1*strides[0] + f2*strides[1] + f3*strides[2] + f4];
            set_tei_J(bf1+f1,bf2+f2,bf3+f3,bf4+f4,value);
            set_tei_K(bf1+f1,bf4+f4,bf3+f3,bf2+f2,value);
        }
    }
}

void LibintInterface::compute_nuclear_potential() 
{
    // Initialise value
    double vnuc = 0.0;

    // Loop over atom pairs
    for(size_t i=0;   i < m_mol.atoms.size(); i++)
    for(size_t j=i+1; j < m_mol.atoms.size(); j++)
    {
        auto xij = m_mol.atoms[i].x - m_mol.atoms[j].x;
        auto yij = m_mol.atoms[i].y - m_mol.atoms[j].y;
        auto zij = m_mol.atoms[i].z - m_mol.atoms[j].z;
        auto r2 = xij * xij + yij * yij + zij * zij;
        auto r = sqrt(r2);
        vnuc += m_mol.atoms[i].atomic_number * m_mol.atoms[j].atomic_number / r;
    }

    // Set the value
    set_scalar_potential(vnuc);
}


void LibintInterface::compute_overlap()
{
    // Get information about shells
    auto shell2bf = m_basis.shell2bf();

    // Resize and zero the overlap matrix
    m_S.resize(m_nbsf*m_nbsf);
    std::fill(m_S.begin(), m_S.end(), 0.0);

    // Evaluate overlap matrix elements
    Engine ov_engine(Operator::overlap, m_basis.max_nprim(), m_basis.max_l());
    const auto &buf_ov = ov_engine.results();
    for(size_t s1=0; s1 < m_basis.size(); ++s1)
    for(size_t s2=0; s2 < m_basis.size(); ++s2)
    {
        // Compute values for this shell set
        ov_engine.compute(m_basis[s1], m_basis[s2]);
        auto *ints = buf_ov[0];
        if(ints == nullptr) // Skip if all integrals screened out
            continue; 

        // Save values for this shell set
        size_t bf1 = shell2bf[s1];
        size_t bf2 = shell2bf[s2];
        size_t n1 = m_basis[s1].size();
        size_t n2 = m_basis[s2].size(); 
        for(size_t f1=0; f1 < n1; ++f1)
        for(size_t f2=0; f2 < n2; ++f2)
        {   
            // Save the value
            set_ovlp(bf1+f1, bf2+f2, ints[f1*n2+f2]);
        }
    }
}

void LibintInterface::compute_orthogonalization_matrix()
{
    // Compute the orthogonalisation matrix
    size_t dim = m_nbsf;
    m_nmo = orthogonalisation_matrix(m_nbsf, m_S, 1.0e-8, m_X);
}


void LibintInterface::compute_one_electron_matrix()
{
    // Setup S dimensions
    auto shell2bf = m_basis.shell2bf();

    // Resize and zero vector storage
    m_oei_a.resize(m_nbsf*m_nbsf);
    m_oei_b.resize(m_nbsf*m_nbsf);
    std::fill(m_oei_a.begin(), m_oei_a.end(), 0.0);
    std::fill(m_oei_b.begin(), m_oei_b.end(), 0.0);

    // Setup kinetic energy engine
    Engine kin_engine(Operator::kinetic, m_basis.max_nprim(), m_basis.max_l());
    // Setup nuclear attraction engine
    Engine nuc_engine(Operator::nuclear, m_basis.max_nprim(), m_basis.max_l());
    nuc_engine.set_params(make_point_charges(m_mol.atoms));
    // Memory buffers
    const auto &buf_kin = kin_engine.results();
    const auto &buf_nuc = nuc_engine.results();

    // Loop over shell pairs
    for(size_t s1=0; s1 < m_basis.size(); ++s1)
    for(size_t s2=0; s2 < m_basis.size(); ++s2)
    {
        // Compute values for this shell set
        kin_engine.compute(m_basis[s1], m_basis[s2]);
        nuc_engine.compute(m_basis[s1], m_basis[s2]);
        auto *ke_ints = buf_kin[0];
        auto *pe_ints = buf_nuc[0];    
        
        // Save values for this shell set
        size_t bf1 = shell2bf[s1];
        size_t bf2 = shell2bf[s2];
        size_t n1 = m_basis[s1].size();
        size_t n2 = m_basis[s2].size(); 
        for(size_t f1=0; f1 < n1; ++f1)
        for(size_t f2=0; f2 < n2; ++f2)
        {
            // Extract the value from the integral buffers
            double value = 0.0;
            if(not (ke_ints == nullptr)) 
                value += ke_ints[f1*n2+f2];
            if(not (pe_ints == nullptr)) 
                value += pe_ints[f1*n2+f2];
       
            // Set alfa and beta term
            set_oei(bf1+f1, bf2+f2, value, true);
            set_oei(bf1+f1, bf2+f2, value, false);
        }
    }
} 

void LibintInterface::build_fock(std::vector<double> &dens, std::vector<double> &fock)
{
    size_t n2 = m_nbsf * m_nbsf;
    // Check dimensions of density matrix
    assert(dens.size() == n2);

    // First get JK
    build_JK(dens,fock);

    // Then add one-electron
    # pragma omp_parallel for
    for(size_t pq=0; pq < n2; pq++)
        fock[pq] += m_oei_a[pq];
}

void LibintInterface::build_JK(std::vector<double> &dens, std::vector<double> &JK)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;
    // Check dimensions of density matrix
    assert(dens.size() == n2);

    // Resize JK matrix
    JK.resize(n2);
    std::fill(JK.begin(),JK.end(),0.0);

    std::vector<double> J(n2), K(n2);
    std::fill(J.begin(),J.end(),0.0);
    std::fill(K.begin(),K.end(),0.0);

    // Loop over basis functions
    // TODO: HGAB 30/08/2024
    //       Need to add OMP memory management here so we can loop over pqrs values freely 
    //       and accumulate the result from each thresd
    //for(size_t p=0; p<m_nbsf; p++)
    //for(size_t q=p; q<m_nbsf; q++)
    //for(size_t r=0; r<m_nbsf; r++)
    //for(size_t s=r; s<m_nbsf; s++)
    //{
    //    size_t pq = p*m_nbsf+q;
    //    size_t rs = r*m_nbsf+s;
    //    J[p*m_nbsf+q] += dens[r*m_nbsf+s] * m_tei_J[pq*n2+rs];
    //    if(p!=q)
    //        J[q*m_nbsf+p] += dens[r*m_nbsf+s] * m_tei_J[pq*n2+rs];
    //    if(r!=s)
    //        J[p*m_nbsf+q] += dens[s*m_nbsf+r] * m_tei_J[pq*n2+rs];
    //    if(r!=s and p!=q)
    //        J[q*m_nbsf+p] += dens[s*m_nbsf+r] * m_tei_J[pq*n2+rs];
    //}

    #pragma omp parallel for
    for(size_t pq=0; pq < n2; pq++)
        for(size_t sr=0; sr < n2; sr++)
            JK[pq] += dens[sr] * (2.0 * m_tei_J[pq*n2+sr] - m_tei_K[pq*n2+sr]);
}


void LibintInterface::build_J(std::vector<double> &dens, std::vector<double> &J)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;
    // Check dimensions of density matrix
    assert(dens.size() == n2);

    // Resize JK matrix
    J.resize(n2);
    std::fill(J.begin(),J.end(),0.0);

    // Loop over basis functions
    #pragma omp parallel for
    for(size_t pq=0; pq < n2; pq++)
        for(size_t sr=0; sr < n2; sr++)
            J[pq] += dens[sr] * m_tei_J[pq*n2+sr];
}

void LibintInterface::build_multiple_JK(
    std::vector<double> &DJ, std::vector<double> &vDK, 
    std::vector<double> &J, std::vector<double> &vK, size_t nk)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;

    // Check dimensions of density matrix
    assert(DJ.size() == n2);
    // Check dimensions of exchange matrices
    assert(vDK.size() == nk * n2);

    // Resize J matrix
    J.resize(n2);
    std::fill(J.begin(),J.end(),0.0);
    // Resize K matrices
    vK.resize(nk * n2);
    std::fill(vK.begin(),vK.end(),0.0);

    // Loop over basis functions
    #pragma omp parallel for 
    for(size_t pq=0; pq < n2; pq++)
        for(size_t sr=0; sr < n2; sr++)
        {
            J[pq] += DJ[sr] * m_tei_J[pq*n2+sr];
            for(size_t k=0; k < nk; k++)
                vK[k*n2+pq] += vDK[k*n2+sr] * m_tei_K[pq*n2+sr];
        }
}


void LibintInterface::tei_ao_to_mo(
    std::vector<double> &C1, std::vector<double> &C2, 
    std::vector<double> &C3, std::vector<double> &C4, 
    std::vector<double> &eri, bool alpha1, bool alpha2)
{
    // Tolerance for screening
    double tol = 1e-14;

    size_t n2 = m_nbsf * m_nbsf;

    // Check dimensions
    assert(C1.size() % m_nbsf == 0);
    assert(C2.size() % m_nbsf == 0);
    assert(C3.size() % m_nbsf == 0);
    assert(C4.size() % m_nbsf == 0);
    
    // Get number of columns of transformation matrices
    size_t d1 = C1.size() / m_nbsf;
    size_t d2 = C2.size() / m_nbsf;
    size_t d3 = C3.size() / m_nbsf;
    size_t d4 = C4.size() / m_nbsf;

    // Define temporary memory
    std::vector<double> tmp1(n2*n2, 0.0);
    std::vector<double> tmp2(n2*n2, 0.0);    

    // Set tmp2 to the antisymmetrised values as appropriate and convert chemist to physicist
    double scale = (alpha1 == alpha2) ? 1.0 : 0.0; 
    #pragma omp parallel for collapse(4)
    for(size_t mu=0; mu < m_nbsf; mu++)
    for(size_t nu=0; nu < m_nbsf; nu++)
    for(size_t sg=0; sg < m_nbsf; sg++)
    for(size_t ta=0; ta < m_nbsf; ta++)
    {
        size_t i1 = (mu*m_nbsf+nu)*n2 + sg*m_nbsf+ta;
        size_t i2 = (mu*m_nbsf+sg)*n2 + nu*m_nbsf+ta;
        // <mn||st> = J[m,s,n,t] - K[m,s,n,t]
        tmp2[i1] = m_tei_J[i2] - scale * m_tei_K[i2];
    }

 
    // Transform s index
    #pragma omp parallel for collapse(2)
    for(size_t mu=0; mu < m_nbsf; mu++)
    for(size_t nu=0; nu < m_nbsf; nu++)
    {
        // Define memory buffers
        double *buff1 = &tmp2[mu*m_nbsf*m_nbsf*m_nbsf + nu*m_nbsf*m_nbsf];
        double *buff2 = &tmp1[mu*m_nbsf*m_nbsf*d4 + nu*m_nbsf*d4];
        // Perform inner loop
        for(size_t sg=0; sg < m_nbsf; sg++)
        for(size_t ta=0; ta < m_nbsf; ta++)
        {
            // Get integral value
            double Ivalue = buff1[sg*m_nbsf + ta];
            // Skip if integral is (near) zero
            if(std::abs(Ivalue) < tol) 
                continue;
            // Add contribution to temporary array
            for(size_t s=0; s < d4; s++)
                buff2[sg*d4 + s] += Ivalue * C4[ta*d4 + s];
        }
    } 

    // Transform r index
    std::fill(tmp2.begin(), tmp2.end(), 0.0);
    #pragma omp parallel for collapse(2)
    for(size_t mu=0; mu < m_nbsf; mu++)
    for(size_t nu=0; nu < m_nbsf; nu++)
    {
        // Define memory buffers
        double *buff1 = &tmp1[mu*m_nbsf*m_nbsf*d4 + nu*m_nbsf*d4];
        double *buff2 = &tmp2[mu*m_nbsf*d3*d4 + nu*d3*d4];
        // Perform inner loop
        for(size_t sg=0; sg < m_nbsf; sg++)
        for(size_t s=0; s < d4; s++)
        {
            // Get integral value
            double Ivalue = buff1[sg*d4 + s];
            // Skip if integral is (near) zero
            if(std::abs(Ivalue) < tol) 
                continue;   
            // Add contribution to temporary array
            for(size_t r=0; r < d3; r++)
                buff2[r*d4 + s] += Ivalue * C3[sg*d3 + r];
        }
    }

    // Transform q index. 
    std::fill(tmp1.begin(), tmp1.end(), 0.0);
    #pragma omp parallel for collapse(2)
    for(size_t mu=0; mu < m_nbsf; mu++)
    for(size_t q=0; q < d2; q++)
    {
        // Define memory buffers
        double *buff1 = &tmp2[mu*m_nbsf*d3*d4];
        double *buff2 = &tmp1[mu*d2*d3*d4+q*d3*d4];

        // Perform inner loop
        for(size_t nu=0; nu < m_nbsf; nu++)
        for(size_t r=0; r < d3; r++)
        for(size_t s=0; s < d4; s++)
        {
            // Get integral value
            double Ivalue = buff1[nu*d3*d4 + r*d4 + s];
            // Skip if integral is (near) zero
            if(std::abs(Ivalue) < tol) 
                continue;
            // Add contribution to temporary array
            buff2[r*d4 + s] += Ivalue * C2[nu*d2 + q];
        }
    }

    // Transform p index
    eri.resize(d1*d2*d3*d4);
    std::fill(eri.begin(), eri.end(), 0.0);
    #pragma omp parallel for collapse(2)
    for(size_t p=0; p < d1; p++)
    for(size_t q=0; q < d2; q++)
    {
        // Define memory buffers
        double *buff1 = &eri[p*d2*d3*d4 + q*d3*d4];
        for(size_t mu=0; mu < m_nbsf; mu++)
        {
            double *buff2 = &tmp1[mu*d2*d3*d4];
            // Perform inner loop
            for(size_t r=0; r < d3; r++)
            for(size_t s=0; s < d4; s++)
            {
                // Get integral value
                double Ivalue = buff2[q*d3*d4 + r*d4 + s];
                // Skip if integral is (near) zero
                if(std::abs(Ivalue) < tol) 
                    continue;
                // Add contribution to output array
                buff1[r*d4 + s] += Ivalue * C1[mu*d1 + p];
            }
        }
    }
}

void LibintInterface::oei_ao_to_mo(
    std::vector<double> &C1, std::vector<double> &C2, 
    std::vector<double> &oei_mo, bool alpha)
{
    // Check dimensions
    assert(C1.size() % m_nbsf == 0);
    assert(C2.size() % m_nbsf == 0);

    // Get number of columns of transformation matrices
    size_t d1 = C1.size() / m_nbsf;
    size_t d2 = C2.size() / m_nbsf;

    // Get alfa or beta one-electron integrals
    std::vector<double> &oei = alpha ? m_oei_a : m_oei_b;

    // Perform transformation
    oei_transform(C1,C2,oei,oei_mo,d1,d2,m_nbsf);
}
