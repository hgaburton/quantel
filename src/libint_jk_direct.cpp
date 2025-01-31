#include <omp.h>
#include <fstream>
#include <libint2/lcao/molden.h>
#include <Eigen/Dense>
#include "libint_interface.h"
#include "linalg.h"

using namespace libint2;
using namespace Eigen;

void LibintInterface::direct_JK(std::vector<double> &dens, std::vector<double> &JK)
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

void LibintInterface::direct_multiple_JK(
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


void LibintInterface::direct_tei_ao_to_mo(
    std::vector<double> &C1, std::vector<double> &C2, 
    std::vector<double> &C3, std::vector<double> &C4, 
    std::vector<double> &eri, bool alpha1, bool alpha2)
{
    throw std::runtime_error("LibintInterface::direct_tei_ao_to_mo() not implemented");
}
