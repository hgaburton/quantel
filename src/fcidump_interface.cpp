#include <omp.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cctype>
#include "fcidump_interface.h"
#include "linalg.h"
#include "omp_device.h"

void FCIDumpInterface::initialize(const std::string &filename)
{
    parse_fcidump(filename);
}

namespace {

/// Return an uppercase copy of s
std::string to_upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

/// Parse the integer value for key KEY= from a header string (upper-case).
/// Returns -1 when the key is absent.
long parse_header_int(const std::string &upper_header, const std::string &key)
{
    std::string tok = key + "=";
    size_t pos = upper_header.find(tok);
    if (pos == std::string::npos) return -1;
    pos += tok.size();
    // Skip optional whitespace
    while (pos < upper_header.size() && std::isspace(upper_header[pos])) ++pos;
    // Collect digits (and optional leading minus)
    std::string numstr;
    if (pos < upper_header.size() && upper_header[pos] == '-') numstr += upper_header[pos++];
    while (pos < upper_header.size() && std::isdigit(upper_header[pos]))
        numstr += upper_header[pos++];
    return numstr.empty() ? -1 : std::stol(numstr);
}

} // anonymous namespace

void FCIDumpInterface::parse_fcidump(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("FCIDumpInterface: cannot open file '" + filename + "'");

    // -----------------------------------------------------------------------
    // 1. Read the header block
    // -----------------------------------------------------------------------
    std::string line;
    std::string header;          // accumulated header text (stripped of &FCI / &END)
    bool in_header = false;

    while (std::getline(file, line))
    {
        std::string upper = to_upper(line);

        if (!in_header)
        {
            if (upper.find("&FCI") != std::string::npos)
            {
                in_header = true;
                // Keep the part after &FCI on the same line
                size_t pos = upper.find("&FCI") + 4;
                header += line.substr(pos) + " ";
            }
            continue;
        }

        // Check for end-of-header markers
        // '/' alone on a line (possibly surrounded by whitespace) or &END anywhere
        std::string stripped = line;
        stripped.erase(
            std::remove_if(stripped.begin(), stripped.end(), ::isspace),
            stripped.end());

        if (stripped == "/" || upper.find("&END") != std::string::npos)
        {
            in_header = false;
            break;
        }

        header += line + " ";
    }

    if (in_header)
        throw std::runtime_error("FCIDumpInterface: end of header (&END or /) not found in '" + filename + "'");

    // -----------------------------------------------------------------------
    // 2. Parse NORB, NELEC, MS2 from header
    // -----------------------------------------------------------------------
    std::string upper_header = to_upper(header);

    long norb  = parse_header_int(upper_header, "NORB");
    long nelec = parse_header_int(upper_header, "NELEC");
    long ms2   = parse_header_int(upper_header, "MS2");

    if (norb  <= 0) throw std::runtime_error("FCIDumpInterface: NORB not found or invalid");
    if (nelec < 0)  throw std::runtime_error("FCIDumpInterface: NELEC not found or invalid");

    m_nbsf  = static_cast<size_t>(norb);
    m_nmo   = m_nbsf;
    m_nelec = static_cast<size_t>(nelec);
    m_ms2   = (ms2 >= 0) ? static_cast<size_t>(ms2) : 0;

    // -----------------------------------------------------------------------
    // 3. Allocate and initialise storage
    // -----------------------------------------------------------------------
    size_t n2 = m_nbsf * m_nbsf;
    size_t n4 = n2 * n2;

    m_V = 0.0;

    m_S.assign(n2, 0.0);
    m_X.assign(n2, 0.0);
    for (size_t p = 0; p < m_nbsf; p++)
    {
        m_S[p*m_nbsf+p] = 1.0;
        m_X[p*m_nbsf+p] = 1.0;
    }

    m_oei_a.assign(n2, 0.0);
    m_oei_b.assign(n2, 0.0);
    m_tei.assign(n4, 0.0);

    // -----------------------------------------------------------------------
    // 4. Read integral data lines
    // -----------------------------------------------------------------------
    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        // Fortran scientific notation: replace D/d exponent markers with E/e
        std::replace(line.begin(), line.end(), 'D', 'E');
        std::replace(line.begin(), line.end(), 'd', 'e');

        std::istringstream iss(line);
        double value;
        long   i, j, k, l;

        if (!(iss >> value >> i >> j >> k >> l)) continue;

        if (i == 0 && j == 0 && k == 0 && l == 0)
        {
            // Scalar potential (nuclear repulsion or frozen-core energy)
            m_V = value;
        }
        else if (k == 0 && l == 0)
        {
            // One-electron integral h(i,j) — 1-indexed, symmetric
            size_t pi = static_cast<size_t>(i) - 1;
            size_t pj = static_cast<size_t>(j) - 1;
            m_oei_a[pi*m_nbsf+pj] = value;
            m_oei_a[pj*m_nbsf+pi] = value;
            m_oei_b[pi*m_nbsf+pj] = value;
            m_oei_b[pj*m_nbsf+pi] = value;
        }
        else
        {
            // Save value in chemists notation and apply 8 fold symmetry (ij|kl)=<ik|jl>
            size_t pi = static_cast<size_t>(i) - 1;
            size_t pj = static_cast<size_t>(j) - 1;
            size_t pk = static_cast<size_t>(k) - 1;
            size_t pl = static_cast<size_t>(l) - 1;

            m_tei[tei_index(pi,pj,pk,pl)] = value;
            m_tei[tei_index(pj,pi,pk,pl)] = value;
            m_tei[tei_index(pi,pj,pl,pk)] = value;
            m_tei[tei_index(pj,pi,pl,pk)] = value;
            m_tei[tei_index(pk,pl,pi,pj)] = value;
            m_tei[tei_index(pl,pk,pi,pj)] = value;
            m_tei[tei_index(pk,pl,pj,pi)] = value;
            m_tei[tei_index(pl,pk,pj,pi)] = value;
        }
    }
}

void FCIDumpInterface::build_fock(
    std::vector<double> &dens, std::vector<double> &fock)
{
    size_t n2 = m_nbsf * m_nbsf;
    assert(dens.size() == n2);

    build_JK(dens, fock);

    #pragma omp parallel for
    for (size_t pq = 0; pq < n2; pq++)
        fock[pq] += m_oei_a[pq];
}

void FCIDumpInterface::build_JK(
    std::vector<double> &dens, std::vector<double> &JK)
{
    size_t n2 = m_nbsf * m_nbsf;
    assert(dens.size() == n2);

    JK.resize(n2);
    std::fill(JK.begin(), JK.end(), 0.0);

    omp_device dev;
    std::vector<double> Jt(dev.nthreads*n2), Kt(dev.nthreads*n2);
    std::fill(Jt.begin(),Jt.end(),0.0);
    std::fill(Kt.begin(),Kt.end(),0.0);

     
    // Compute J matrix elements (need to repeat for each density)
    // Jpq = (pq|rs) * Dsr = \sum_rs <pr|qs> * Dsr
    // Kps = (pq|rs) * Dqr = \sum_rs <pr|sq> * Dsr
    #pragma omp parallel for collapse(2) schedule(guided)
    for (size_t p=0; p<m_nbsf; p++)
    for (size_t r=0; r<m_nbsf; r++)
    {
        int ithread = dev.thread_id();
        double *J = &Jt[ithread*n2];
        double *K = &Kt[ithread*n2];

        for (size_t q=p; q<m_nbsf; q++)
        for (size_t s=r; s<m_nbsf; s++)
        {
            size_t pq = p*m_nbsf+q;
            size_t rs = r*m_nbsf+s;
            if (pq > rs) continue;
            
            double Vpqrs = m_tei[pq*n2+rs];
            if (std::abs(Vpqrs) < thresh) continue;

            J[p*m_nbsf+q] += dens[s*m_nbsf+r] * Vpqrs;
            K[p*m_nbsf+s] += dens[q*m_nbsf+r] * Vpqrs;
            if (p != q) {
                J[q*m_nbsf+p] += dens[s*m_nbsf+r] * Vpqrs;
                K[q*m_nbsf+s] += dens[p*m_nbsf+r] * Vpqrs;
            }
            if (r != s) {
                J[p*m_nbsf+q] += dens[r*m_nbsf+s] * Vpqrs;
                K[p*m_nbsf+r] += dens[q*m_nbsf+s] * Vpqrs;
            }
            if (r != s && p != q) {
                J[q*m_nbsf+p] += dens[r*m_nbsf+s] * Vpqrs;
                K[q*m_nbsf+r] += dens[p*m_nbsf+s] * Vpqrs;
            }

            if (pq != rs)
            {
                J[r*m_nbsf+s] += dens[q*m_nbsf+p] * Vpqrs;
                K[r*m_nbsf+q] += dens[s*m_nbsf+p] * Vpqrs;
                if (r != s) {
                    J[s*m_nbsf+r] += dens[q*m_nbsf+p] * Vpqrs;
                    K[s*m_nbsf+q] += dens[r*m_nbsf+p] * Vpqrs;
                }
                if (p != q) {
                    J[r*m_nbsf+s] += dens[p*m_nbsf+q] * Vpqrs;
                    K[r*m_nbsf+p] += dens[s*m_nbsf+q] * Vpqrs;
                }
                if (r != s && p != q) {
                    J[s*m_nbsf+r] += dens[p*m_nbsf+q] * Vpqrs;
                    K[s*m_nbsf+p] += dens[r*m_nbsf+q] * Vpqrs;
                }
            }
        }
    }

    // Consolidate date from different threads
    for (size_t it=0; it<dev.nthreads; it++)
    for (size_t pq=0; pq<n2; pq++)
        JK[pq] += 2.0 * Jt[it*n2+pq] - Kt[it*n2+pq];
}

void FCIDumpInterface::build_multiple_JK(
    std::vector<double> &vDJ, std::vector<double> &vDK,
    std::vector<double> &vJ,  std::vector<double> &vK,
    size_t nj, size_t nk)
{
    // Get size of n2 for indexing later
    size_t n2 = m_nbsf * m_nbsf;

    // Check dimensions of density matrix
    assert(vDJ.size() == nj * n2);
    // Check dimensions of exchange matrices
    assert(vDK.size() == nk * n2);

    // Resize J matrix
    vJ.resize(nj*n2);
    std::fill(vJ.begin(),vJ.end(),0.0);
    // Resize K matrices
    vK.resize(nk*n2);
    std::fill(vK.begin(),vK.end(),0.0);

    // Setup thread-safe memory
    omp_device dev;
    std::vector<double> Jsafe(dev.nthreads*n2*nj), Ksafe(dev.nthreads*n2*nk);
    std::fill(Jsafe.begin(),Jsafe.end(),0.0);
    std::fill(Ksafe.begin(),Ksafe.end(),0.0);

    // Loop over basis functions
    #pragma omp parallel for collapse(2) schedule(guided)
    for(size_t p=0; p<m_nbsf; p++)
    for(size_t r=0; r<m_nbsf; r++)
    {
        int ithread = dev.thread_id();
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
            // Jpq = (pq|rs) * Dsr = \sum_rs <pr|qs> * Dsr
            for(size_t k=0; k < nj; k++)
            {
                double *Jt_k = &Jt[k*n2];
                double *Dj   = &vDJ[k*n2];
                Jt_k[p*m_nbsf+q] += Dj[s*m_nbsf+r] * Vpqrs;
                if(p!=q) Jt_k[q*m_nbsf+p] += Dj[s*m_nbsf+r] * Vpqrs;
                if(r!=s) Jt_k[p*m_nbsf+q] += Dj[r*m_nbsf+s] * Vpqrs;
                if(r!=s and p!=q) Jt_k[q*m_nbsf+p] += Dj[r*m_nbsf+s] * Vpqrs;

                if(pq != rs) 
                {
                    Jt_k[r*m_nbsf+s] += Dj[q*m_nbsf+p] * Vpqrs;
                    if(r!=s) Jt_k[s*m_nbsf+r] += Dj[q*m_nbsf+p] * Vpqrs;
                    if(p!=q) Jt_k[r*m_nbsf+s] += Dj[p*m_nbsf+q] * Vpqrs;
                    if(r!=s and p!=q) Jt_k[s*m_nbsf+r] += Dj[p*m_nbsf+q] * Vpqrs;
                }                
            }

            // Compute K matrix elements (need to repeat for each density)
            // Kps = (pq|rs) * Dqr = \sum_rs <pr|sq> * Dsr
            for(size_t k=0; k < nk; k++)
            {
                double *Kt_k = &Kt[k*n2];
                double *Dk   = &vDK[k*n2];
                Kt_k[p*m_nbsf+s] += Dk[q*m_nbsf+r] * Vpqrs;
                if(p!=q) Kt_k[q*m_nbsf+s] += Dk[p*m_nbsf+r] * Vpqrs;
                if(r!=s) Kt_k[p*m_nbsf+r] += Dk[q*m_nbsf+s] * Vpqrs;
                if(r!=s and p!=q) Kt_k[q*m_nbsf+r] += Dk[p*m_nbsf+s] * Vpqrs;

                if(pq != rs) 
                {
                    Kt_k[r*m_nbsf+q] += Dk[s*m_nbsf+p] * Vpqrs;
                    if(r!=s) Kt_k[s*m_nbsf+q] += Dk[r*m_nbsf+p] * Vpqrs;
                    if(p!=q) Kt_k[r*m_nbsf+p] += Dk[s*m_nbsf+q] * Vpqrs;
                    if(r!=s and p!=q) Kt_k[s*m_nbsf+p] += Dk[r*m_nbsf+q] * Vpqrs;
                }
            }
        }
    }

    // Collect values for each thread
    for(size_t it=0; it < dev.nthreads; it++)
    {
        for(size_t pqk=0; pqk < n2 * nj; pqk++)
            vJ[pqk] += Jsafe[it*n2*nj+pqk];
        for(size_t pqk=0; pqk < n2 * nk; pqk++)
            vK[pqk] += Ksafe[it*n2*nk+pqk];
    }
}

void FCIDumpInterface::oei_ao_to_mo(
    std::vector<double> &C1, std::vector<double> &C2,
    std::vector<double> &oei_mo, bool alpha)
{
    assert(C1.size() % m_nbsf == 0);
    assert(C2.size() % m_nbsf == 0);

    size_t d1 = C1.size() / m_nbsf;
    size_t d2 = C2.size() / m_nbsf;

    std::vector<double> &oei = alpha ? m_oei_a : m_oei_b;
    oei_transform(C1, C2, oei, oei_mo, d1, d2, m_nbsf);
}

void FCIDumpInterface::tei_ao_to_mo(
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

    // Get array dimensions
    size_t dim = (std::max(m_nbsf,d1) * std::max(m_nbsf,d2) *
                  std::max(m_nbsf,d3) * std::max(m_nbsf,d4));

    // Define temporary memory
    std::vector<double> tmp1(dim, 0.0);
    std::vector<double> tmp2(dim, 0.0);    

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


MOintegrals FCIDumpInterface::mo_integrals(
    std::vector<double> &C, size_t ncore, size_t nactive)
{
    if (nactive == 0) nactive = m_nmo - ncore;

    if (C.size() == 0)
        throw std::runtime_error("FCIDumpInterface::mo_integrals: no orbital coefficients provided");
    if (C.size() != m_nbsf * m_nmo)
        throw std::runtime_error("FCIDumpInterface::mo_integrals: orbital coefficients have wrong dimensions");
    if (ncore + nactive > m_nmo)
        throw std::runtime_error("FCIDumpInterface::mo_integrals: invalid number of orbitals");

    // Extract active orbital coefficients C_act (m_nbsf x nactive)
    std::vector<double> Cact(m_nbsf * nactive, 0.0);
    #pragma omp parallel for collapse(2)
    for (size_t mu = 0; mu < m_nbsf; mu++)
    for (size_t p  = 0; p  < nactive; p++)
        Cact[mu*nactive+p] = C[mu*m_nmo+(p+ncore)];

    // Core density matrix
    std::vector<double> Pcore(m_nbsf * m_nbsf, 0.0);
    #pragma omp parallel for collapse(2)
    for (size_t p = 0; p < m_nbsf; p++)
    for (size_t q = 0; q < m_nbsf; q++)
        for (size_t i = 0; i < ncore; i++)
            Pcore[p*m_nbsf+q] += C[p*m_nmo+i] * C[q*m_nmo+i];

    // Scalar core energy and effective one-electron potential
    std::vector<double> Veff(nactive * nactive, 0.0);
    double Vmo = scalar_potential();
    if (ncore > 0)
    {
        std::vector<double> JK(m_nbsf * m_nbsf, 0.0);
        build_JK(Pcore, JK);

        double *Hao = oei_matrix(true);
        #pragma omp parallel for reduction(+:Vmo)
        for (size_t pq = 0; pq < m_nbsf * m_nbsf; pq++)
            Vmo += 2.0 * (Hao[pq] + 0.5 * JK[pq]) * Pcore[pq];

        oei_transform(Cact, Cact, JK, Veff, nactive, nactive, m_nbsf);
    }

    // One-electron integrals in active space
    std::vector<double> h1e_mo(nactive * nactive, 0.0);
    oei_ao_to_mo(Cact, Cact, h1e_mo, true);
    if (ncore > 0)
        for (size_t pq = 0; pq < nactive * nactive; pq++)
            h1e_mo[pq] += Veff[pq];

    // Two-electron integrals in active space
    std::vector<double> eri_mo;
    tei_ao_to_mo(Cact, Cact, Cact, Cact, eri_mo, true, false);

    return MOintegrals(Vmo, h1e_mo, eri_mo, nactive);
}
