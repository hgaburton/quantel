#include "libint_interface.h"
#include "linalg.h"

using namespace libint2;


void LibintInterface::initialize()
{
    // Save number of basis functions
    m_nbsf = m_basis.size();

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

double LibintInterface::tei(size_t p, size_t q, size_t r, size_t s, bool alpha1, bool alpha2) 
{
    size_t index = tei_index(p,q,r,s);
    if(alpha1 == true and alpha2 == true)
        return m_tei_aa[index];
    if(alpha1 == true and alpha2 == false)
        return m_tei_ab[index];
    if(alpha1 == false and alpha2 == false)    
        return m_tei_bb[index];
    
    return 0;
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

void LibintInterface::set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1, bool alpha2)
{
    size_t index = tei_index(p,q,r,s);
    if(alpha1 == true and alpha2 == true)
        m_tei_aa[index] = value;
    if(alpha1 == true and alpha2 == false)
        m_tei_ab[index] = value;
    if(alpha1 == false and alpha2 == false)    
        m_tei_bb[index] = value;
}


void LibintInterface::compute_two_electron_integrals()
{
    // Setup II dimensions
    auto shell2bf = m_basis.shell2bf();
    size_t strides[3];

    // Setup array dimensions
    size_t dim = m_nbsf * m_nbsf * m_nbsf * m_nbsf;
    m_tei_aa.resize(dim, 0.0);
    m_tei_ab.resize(dim, 0.0);
    m_tei_bb.resize(dim, 0.0);

    // Compute integrals
    Engine coul_engine(Operator::coulomb, m_basis.max_nprim(), m_basis.max_l());
    const auto &buf_coul = coul_engine.results();
    for(size_t s1=0; s1 < m_nbsf; ++s1)
    for(size_t s2=0; s2 < m_nbsf; ++s2)
    for(size_t s3=0; s3 < m_nbsf; ++s3)
    for(size_t s4=0; s4 < m_nbsf; ++s4)
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

        // Compute contribution
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
            // Save value for aabb block
            double value = ints[f1*strides[0] + f2*strides[1] + f3*strides[2] + f4];
            set_tei(bf1+f1, bf2+f2, bf3+f3, bf4+f4, value, true, false);
        }
    }

    // Compute aaaa and bbbb blocks with anntisymmetrisation
    for(size_t p=0; p < m_nbsf; p++)
    for(size_t q=0; q < m_nbsf; q++)
    for(size_t r=0; r < m_nbsf; r++)
    for(size_t s=0; s < m_nbsf; s++)
    {
        double IIpqrs = tei(p,q,r,s,true,false) - tei(p,q,s,r,true,false);
        set_tei(p,q,r,s,IIpqrs,true,true);
        set_tei(p,q,r,s,IIpqrs,false,false);
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

    // Resize and zero vector storage
    m_S.resize(m_nbsf*m_nbsf, 0.0);

    // Evaluate overlap matrix elements
    Engine ov_engine(Operator::overlap, m_basis.max_nprim(), m_basis.max_l());
    const auto &buf_ov = ov_engine.results();
    for(size_t s1=0; s1 < m_nbsf; ++s1)
    for(size_t s2=0; s2 < m_nbsf; ++s2)
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

    // Resize hcore matrix
    m_oei_a.resize(m_nbsf*m_nbsf, 0.0);
    m_oei_b.resize(m_nbsf*m_nbsf, 0.0);    

    // Setup kinetic energy engine
    Engine kin_engine(Operator::kinetic, m_basis.max_nprim(), m_basis.max_l());
    // Setup nuclear attraction engine
    Engine nuc_engine(Operator::nuclear, m_basis.max_nprim(), m_basis.max_l());
    nuc_engine.set_params(make_point_charges(m_mol.atoms));
    // Memory buffers
    const auto &buf_kin = kin_engine.results();
    const auto &buf_nuc = nuc_engine.results();

    // Loop over shell pairs
    for(size_t s1=0; s1 < m_nbsf; ++s1)
    for(size_t s2=0; s2 < m_nbsf; ++s2)
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
            if( !(ke_ints == nullptr) ) 
                value += ke_ints[f1*n2+f2];
            if( !(pe_ints == nullptr) ) 
                value += pe_ints[f1*n2+f2];
       
            // Set alfa and beta term
            set_oei(bf1+f1, bf2+f2, value, true);
            set_oei(bf1+f1, bf2+f2, value, false);
        }
    }
} 

void LibintInterface::build_fock(std::vector<double> &dens, std::vector<double> &fock)
{
    // Check dimensions of density matrix
    assert(dens.size() == m_nbsf * m_nbsf);

    // Resize fock matrix
    fock.resize(m_nbsf * m_nbsf, 0.0);

    // Loop over basis functions
    for(size_t p=0; p < m_nbsf; p++)
    for(size_t q=0; q < m_nbsf; q++)
    {
        // One-electron contribution
        fock[oei_index(p,q)] += oei(p,q,true);

        // Two-electron contribution
        for(size_t s=0; s < m_nbsf; s++)
        for(size_t r=0; r < m_nbsf; r++)
        {
            // Build fock matrix
            fock[oei_index(p,q)] += dens[oei_index(s,r)] * (2.0 * tei(p,q,r,s,true,false) - tei(p,q,s,r,true,false));
        }
    }
}

void LibintInterface::build_JK(std::vector<double> &dens, std::vector<double> &JK)
{
    // Check dimensions of density matrix
    assert(dens.size() == m_nbsf * m_nbsf);

    // Resize JK matrix
    JK.resize(m_nbsf * m_nbsf, 0.0);

    // Loop over basis functions
    for(size_t p=0; p < m_nbsf; p++)
    for(size_t q=0; q < m_nbsf; q++)
    {
        // Two-electron contribution
        for(size_t s=0; s < m_nbsf; s++)
        for(size_t r=0; r < m_nbsf; r++)
        {
            // Build JK matrix
            JK[oei_index(p,q)] += dens[oei_index(s,r)] * (2.0 * tei(p,q,r,s,true,false) - tei(p,q,s,r,true,false));
        }
    }
}