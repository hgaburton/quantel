#include <omp.h>
#include <fstream>
#include <libint2/lcao/molden.h>
#include "libint_interface.h"
#include "linalg.h"

using namespace libint2;
using namespace Eigen;

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

    // Compute the dipole integrals
    compute_dipole_integrals();
}

void LibintInterface::compute_two_electron_integrals()
{
    // Setup II dimensions
    auto shell2bf = m_basis.shell2bf();
    size_t strides[3];

    // Resize and zero the two-electron integral arrays
    m_tei.resize(m_nbsf*m_nbsf*m_nbsf*m_nbsf);
    std::fill(m_tei.begin(), m_tei.end(), 0.0);

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
            set_tei(bf1+f1,bf2+f2,bf3+f3,bf4+f4,value);
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

void LibintInterface::compute_dipole_integrals()
{
    // Compute dipole matrix
    m_dipole.resize(3*m_nbsf*m_nbsf);    
    std::fill(m_dipole.begin(), m_dipole.end(), 0.0);

    // Setup Libint engine
    Engine dip_engine(Operator::emultipole1, m_basis.max_nprim(), m_basis.max_l());
    dip_engine.set_params(std::array<double,3>{0.0,0.0,0.0});

    // Memory buffers
    const auto &buf_vec = dip_engine.results();

    // Loop over shell pairs
    for(size_t s1=0; s1 < m_basis.size(); ++s1)
    for(size_t s2=0; s2 < m_basis.size(); ++s2)
    {
        // Compute values for this shell set
        dip_engine.compute(m_basis[s1], m_basis[s2]);
        auto *dip_ints = buf_vec[0];
        if(dip_ints == nullptr) // Skip if all integrals screened out
            continue; 

        // Save values for this shell set
        auto shell2bf = m_basis.shell2bf();
        size_t bf1 = shell2bf[s1];
        size_t bf2 = shell2bf[s2];
        size_t n1 = m_basis[s1].size();
        size_t n2 = m_basis[s2].size();

        auto s_shellset = buf_vec[0]; // Overlap contribution
        auto mu_x_shellset = buf_vec[2];
        auto mu_y_shellset = buf_vec[3];
        auto mu_z_shellset = buf_vec[1];

        for(size_t f1=0; f1 < n1; ++f1)
        for(size_t f2=0; f2 < n2; ++f2)
        {
            // Get compound p,q index
            size_t pq = (bf1+f1)*m_nbsf+(bf2+f2);
            // Save x terms
            m_dipole[0*m_nbsf*m_nbsf + pq] = mu_x_shellset[f1*n2+f2];
            // Save y terms
            m_dipole[1*m_nbsf*m_nbsf + pq] = mu_y_shellset[f1*n2+f2];
            // Save z terms
            m_dipole[2*m_nbsf*m_nbsf + pq] = mu_z_shellset[f1*n2+f2];
        }
    }
}
    
void LibintInterface::molden_orbs(
    std::vector<double> &C, std::vector<double> &occ, std::vector<double> &evals)
{
    // Convert arrays to Eigen format
    Map<Matrix<double,Dynamic,Dynamic,RowMajor> > coeff(C.data(),m_nbsf,m_nmo);
    Map<VectorXd> mo_occ(occ.data(),m_nmo);
    Map<VectorXd> mo_energy(evals.data(),m_nmo);
    // Export orbitals
    molden::Export xport(m_mol.atoms, m_basis, coeff, mo_occ, mo_energy);
    std::ofstream molden_file("hf++.molden");
    xport.write(molden_file);
}
