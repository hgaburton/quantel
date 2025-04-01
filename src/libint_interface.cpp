#include <omp.h>
#include <fstream>
#include <libint2/lcao/molden.h>
#include <Eigen/Dense>
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
    // Compute the orthogonalisation matrix
    compute_orthogonalization_matrix();
    // Compute the dipole integrals
    compute_dipole_integrals();    
    // Compute eri integrals if in-core
    compute_two_electron_integrals();

    // Setup shell pair list
    std::tie(m_splist, m_spdata) = compute_shellpairs(m_basis);
}

void LibintInterface::compute_nuclear_potential() 
{
    // Initialise value
    double m_Vvnuc = 0.0;

    // Loop over atom pairs
    for(size_t i=0;   i < m_mol.atoms.size(); i++)
    for(size_t j=i+1; j < m_mol.atoms.size(); j++)
    {
        auto xij = m_mol.atoms[i].x - m_mol.atoms[j].x;
        auto yij = m_mol.atoms[i].y - m_mol.atoms[j].y;
        auto zij = m_mol.atoms[i].z - m_mol.atoms[j].z;
        auto r2 = xij * xij + yij * yij + zij * zij;
        auto r = sqrt(r2);
        m_V += m_mol.atoms[i].atomic_number * m_mol.atoms[j].atomic_number / r;
    }
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
            m_S[(bf1+f1)*m_nbsf+bf2+f2] = ints[f1*n2+f2];
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
            m_oei_a[(bf1+f1)*m_nbsf+bf2+f2] = value;
            m_oei_b[(bf1+f1)*m_nbsf+bf2+f2] = value;
        }
    }
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
            m_tei[tei_index(bf1+f1,bf2+f2,bf3+f3,bf4+f4)] = value;
        }
    }
}


void LibintInterface::compute_dipole_integrals()
{
    // Compute dipole matrix
    m_dipole.resize(4*m_nbsf*m_nbsf);    
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
        auto mu_x_shellset = buf_vec[1];
        auto mu_y_shellset = buf_vec[2];
        auto mu_z_shellset = buf_vec[3];

        for(size_t f1=0; f1 < n1; ++f1)
        for(size_t f2=0; f2 < n2; ++f2)
        {
            // Get compound p,q index
            size_t pq = (bf1+f1)*m_nbsf+(bf2+f2);
            // Save overlap term
            m_dipole[pq] = s_shellset[f1*n2+f2];
            // Save x terms
            m_dipole[1*m_nbsf*m_nbsf + pq] = mu_x_shellset[f1*n2+f2];
            // Save y terms
            m_dipole[2*m_nbsf*m_nbsf + pq] = mu_y_shellset[f1*n2+f2];
            // Save z terms
            m_dipole[3*m_nbsf*m_nbsf + pq] = mu_z_shellset[f1*n2+f2];
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


void LibintInterface::build_JK(std::vector<double> &dens, std::vector<double> &JK)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;
    // Check dimensions of density matrix
    assert(dens.size() == n2);

    // Resize JK matrix
    JK.resize(n2);
    std::fill(JK.begin(),JK.end(),0.0);

    // Setup thread-safe memory
    int nthread = omp_get_max_threads();
    std::vector<double> Jt(nthread*n2), Kt(nthread*n2);
    std::fill(Jt.begin(),Jt.end(),0.0);
    std::fill(Kt.begin(),Kt.end(),0.0);

    // Loop over basis functions
    #pragma omp parallel for collapse(2) schedule(guided)
    for(size_t p=0; p<m_nbsf; p++)
    for(size_t r=0; r<m_nbsf; r++)
    {
        int ithread = omp_get_thread_num();
        double *J = &Jt[ithread*n2];
        double *K = &Kt[ithread*n2];

        for(size_t q=p; q<m_nbsf; q++)
        for(size_t s=r; s<m_nbsf; s++)
        {
            size_t pq = p*m_nbsf+q;
            size_t rs = r*m_nbsf+s;
            if(pq > rs) continue;

            // Skip if below threshold
            double Vpqrs = m_tei[pq*n2+rs];
            if(std::abs(Vpqrs) < thresh) continue;

            J[p*m_nbsf+q] += dens[r*m_nbsf+s] * Vpqrs;
            K[p*m_nbsf+s] += dens[r*m_nbsf+q] * Vpqrs;
            if(p!=q) {
                J[q*m_nbsf+p] += dens[r*m_nbsf+s] * Vpqrs;
                K[q*m_nbsf+s] += dens[r*m_nbsf+p] * Vpqrs;
            }
            if(r!=s) {
                J[p*m_nbsf+q] += dens[s*m_nbsf+r] * Vpqrs;
                K[p*m_nbsf+r] += dens[s*m_nbsf+q] * Vpqrs;
            }
            if(r!=s and p!=q) {
                J[q*m_nbsf+p] += dens[s*m_nbsf+r] * Vpqrs;
                K[q*m_nbsf+r] += dens[s*m_nbsf+p] * Vpqrs;
            }

            if(pq != rs) 
            {
                J[r*m_nbsf+s] += dens[p*m_nbsf+q] * Vpqrs;
                K[r*m_nbsf+q] += dens[p*m_nbsf+s] * Vpqrs;
                if(r!=s) {
                    J[s*m_nbsf+r] += dens[p*m_nbsf+q] * Vpqrs;
                    K[s*m_nbsf+q] += dens[p*m_nbsf+r] * Vpqrs;
                }
                if(p!=q) {
                    J[r*m_nbsf+s] += dens[q*m_nbsf+p] * Vpqrs;
                    K[r*m_nbsf+p] += dens[q*m_nbsf+s] * Vpqrs;
                }
                if(r!=s and p!=q) {
                    J[s*m_nbsf+r] += dens[q*m_nbsf+p] * Vpqrs;
                    K[s*m_nbsf+p] += dens[q*m_nbsf+r] * Vpqrs;
                }
            }
        }
    }

    // Collect values for each thread
    for(size_t it=0; it < nthread; it++)
    for(size_t pq=0; pq < n2; pq++)
        JK[pq] += 2.0 * Jt[it*n2+pq] - Kt[it*n2+pq];
}

void LibintInterface::build_multiple_JK(
    std::vector<double> &vDJ, std::vector<double> &vDK,
    std::vector<double> &vJ, std::vector<double> &vK, 
    size_t nj, size_t nk)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;

    // Check dimensions of density matrix
    assert(vDJ.size() == nj * n2);
    // Check dimensions of exchange matrices
    assert(vDK.size() == nk * n2);

    // Resize J matrix
    vJ.resize(nj * n2);
    std::fill(vJ.begin(),vJ.end(),0.0);
    // Resize K matrices
    vK.resize(nk * n2);
    std::fill(vK.begin(),vK.end(),0.0);

    // Setup thread-safe memory
    int nthread = omp_get_max_threads();
    std::vector<double> Jsafe(nthread*n2*nj), Ksafe(nthread*n2*nk);
    std::fill(Jsafe.begin(),Jsafe.end(),0.0);
    std::fill(Ksafe.begin(),Ksafe.end(),0.0);

    // Loop over basis functions
    #pragma omp parallel for collapse(2) schedule(guided)
    for(size_t p=0; p<m_nbsf; p++)
    for(size_t r=0; r<m_nbsf; r++)
    {
        int ithread = omp_get_thread_num();
        double *Jt = &Jsafe[ithread*n2*nj];
        double *Kt = &Ksafe[ithread*n2*nk];

        for(size_t q=p; q<m_nbsf; q++)
        for(size_t s=r; s<m_nbsf; s++)
        {
            size_t pq = p*m_nbsf+q;
            size_t rs = r*m_nbsf+s;
            if(pq > rs) continue;

            // Skip if below threshold
            double Vpqrs = m_tei[pq*n2+rs];
            if(std::abs(Vpqrs) < thresh) continue;

            // Compute J matrix elements (need to repeat for each density)
            for(size_t k=0; k < nj; k++)
            {
                double *Jt_k = &Jt[k*n2];
                double *Dj   = &vDJ[k*n2];
                Jt_k[p*m_nbsf+q] += Dj[r*m_nbsf+s] * Vpqrs;
                if(p!=q) Jt_k[q*m_nbsf+p] += Dj[r*m_nbsf+s] * Vpqrs;
                if(r!=s) Jt_k[p*m_nbsf+q] += Dj[s*m_nbsf+r] * Vpqrs;
                if(r!=s and p!=q) Jt_k[q*m_nbsf+p] += Dj[s*m_nbsf+r] * Vpqrs;

                if(pq != rs) 
                {
                    Jt_k[r*m_nbsf+s] += Dj[p*m_nbsf+q] * Vpqrs;
                    if(r!=s) Jt_k[s*m_nbsf+r] += Dj[p*m_nbsf+q] * Vpqrs;
                    if(p!=q) Jt_k[r*m_nbsf+s] += Dj[q*m_nbsf+p] * Vpqrs;
                    if(r!=s and p!=q) Jt_k[s*m_nbsf+r] += Dj[q*m_nbsf+p] * Vpqrs;
                }                
            }

            // Compute K matrix elements (need to repeat for each density)
            for(size_t k=0; k < nk; k++)
            {
                double *Kt_k = &Kt[k*n2];
                double *Dk   = &vDK[k*n2];
                Kt_k[p*m_nbsf+s] += Dk[r*m_nbsf+q] * Vpqrs;
                if(p!=q) Kt_k[q*m_nbsf+s] += Dk[r*m_nbsf+p] * Vpqrs;
                if(r!=s) Kt_k[p*m_nbsf+r] += Dk[s*m_nbsf+q] * Vpqrs;
                if(r!=s and p!=q) Kt_k[q*m_nbsf+r] += Dk[s*m_nbsf+p] * Vpqrs;

                if(pq != rs) 
                {
                    Kt_k[r*m_nbsf+q] += Dk[p*m_nbsf+s] * Vpqrs;
                    if(r!=s) Kt_k[s*m_nbsf+q] += Dk[p*m_nbsf+r] * Vpqrs;
                    if(p!=q) Kt_k[r*m_nbsf+p] += Dk[q*m_nbsf+s] * Vpqrs;
                    if(r!=s and p!=q) Kt_k[s*m_nbsf+p] += Dk[q*m_nbsf+r] * Vpqrs;
                }
            }
        }
    }

    // Collect values for each thread
    for(size_t it=0; it < nthread; it++)
    {
        for(size_t pqk=0; pqk < n2 * nj; pqk++)
            vJ[pqk] += Jsafe[it*n2*nj+pqk];
        for(size_t pqk=0; pqk < n2 * nk; pqk++)
            vK[pqk] += Ksafe[it*n2*nk+pqk];
    }
}


void LibintInterface::tei_ao_to_mo(
    std::vector<double> &C1, std::vector<double> &C2, 
    std::vector<double> &C3, std::vector<double> &C4, 
    std::vector<double> &eri, bool alpha1, bool alpha2)
{
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
        size_t i3 = (mu*m_nbsf+ta)*n2 + nu*m_nbsf+sg;
        // <mn||st> = (ms|nt) - (mt|ns)
        tmp2[i1] = m_tei[i2] - scale * m_tei[i3];
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
            if(std::abs(Ivalue) < thresh) 
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
            if(std::abs(Ivalue) < thresh) 
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
            if(std::abs(Ivalue) < thresh) 
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
                if(std::abs(Ivalue) < thresh) 
                    continue;
                // Add contribution to output array
                buff1[r*d4 + s] += Ivalue * C1[mu*d1 + p];
            }
        }
    }
}
