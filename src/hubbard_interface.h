#ifndef HUBBARD_INTERFACE_H
#define HUBBARD_INTERFACE_H

#include "integral_interface.h"

class HubbardInterface : public IntegralInterface 
/**
 * @class HubbardInterface
 * @brief Integral interface designed to implement a square 1D/2D/3D Hubbard lattice.
 * 
 * This class inherits from IntegralInterface and provides functionality to 
 * initialize and compute integrals for a Hubbard lattice with specified 
 * dimensions and periodic boundary conditions.
 * 
 * @details
 * The class supports 1D, 2D, and 3D Hubbard lattices with optional periodic 
 * boundary conditions in each dimension. It initializes various matrices 
 * required for computations, including the overlap matrix, hopping matrix, 
 * dipole matrix, and two-electron integrals. The class also provides methods 
 * to compute site indices and orthogonalization matrices.
 * 
 * @note
 * - The number of lattice sites in each dimension must be greater than zero.
 * - Periodic boundary conditions cannot be applied if the number of sites 
 *   in the corresponding dimension is one.
 * 
 * @param U Hubbard U parameter.
 * @param t Hopping parameter.
 * @param nx Number of lattice sites in the x dimension.
 * @param ny Number of lattice sites in the y dimension. Default is 1.
 * @param nz Number of lattice sites in the z dimension. Default is 1.
 * @param periodicX Periodic boundary conditions in the x dimension. Default is false.
 * @param periodicY Periodic boundary conditions in the y dimension. Default is false.
 * @param periodicZ Periodic boundary conditions in the z dimension. Default is false.
 */
{
    /// @brief Integral interface designed to implement a square 1D/2D/3D Hubbard lattice

public:
    /// Destructor for the Hubbard interface
    virtual ~HubbardInterface() { }

    /// @brief Constructor for the Hubbard interface
    /// @param U Hubbard U parameter
    /// @param t hopping parameter
    /// @param na number of alpha electrons
    /// @param nb number of beta electrons
    /// @param nx number of lattice sites in x
    /// @param ny number of lattice sites in y. Default is 1
    /// @param nz number of lattice sites in z. Default is 1
    /// @param periodicX periodic boundary conditions in x. Default is false
    /// @param periodicY periodic boundary conditions in y. Default is false
    /// @param periodicZ periodic boundary conditions in z. Default is false
    HubbardInterface(
        double U, double t, size_t na, size_t nb, 
        std::vector<size_t> dim={1,1,1}, std::vector<bool> periodic={false,false,false}) : m_U(U), m_t(t)
    {
        // Check some input
        if(dim.size() != 3) 
            throw std::runtime_error("Dimension vector must have 3 elements");
        if(periodic.size() != 3) 
            throw std::runtime_error("Periodic vector must have 3 elements");
        
        // Call base class init function
        init(dim[0]*dim[1]*dim[2],na,nb);
        
        // Set Hubbard lattice parameters
        m_nx = dim[0]; m_ny = dim[1]; m_nz = dim[2];
        m_pX = periodic[0]; m_pY = periodic[1]; m_pZ = periodic[2];

        // Check some input
        if(m_nx==0) throw std::runtime_error("Number of sites in X dimension cannot be zero");
        if(m_ny==0) throw std::runtime_error("Number of sites in Y dimension cannot be zero");
        if(m_nz==0) throw std::runtime_error("Number of sites in Z dimension cannot be zero");
        if(m_nx==1 and m_pX) throw std::runtime_error("Cannot have periodic X boundary with 1 site");
        if(m_ny==1 and m_pY) throw std::runtime_error("Cannot have periodic Y boundary with 1 site");
        if(m_nz==1 and m_pZ) throw std::runtime_error("Cannot have periodic Z boundary with 1 site");

        // Useful constants
        size_t n2 = m_nbsf * m_nbsf;

        /// Initialise scalar potential 
        m_V = 0.0;

        /// Initialise hopping matrix
        m_oei_a.resize(m_nbsf*m_nbsf);
        std::fill(m_oei_a.begin(),m_oei_a.end(),0.0);

        /// Initialise overlap matrix
        m_S.resize(m_nbsf*m_nbsf); 
        std::fill(m_S.begin(),m_S.end(),0.0);

        /// Initialise dipole matrix
        m_dipole.resize(3*m_nbsf*m_nbsf);    
        std::fill(m_dipole.begin(), m_dipole.end(), 0.0);
        /// Define centre (assuming lattice has spacing 1)
        double cx = 0.5*(m_nx-1);
        double cy = 0.5*(m_ny-1);
        double cz = 0.5*(m_nz-1);

        /// Initialise two-electron integrals
        m_tei.resize(m_nbsf*m_nbsf*m_nbsf*m_nbsf);
        std::fill(m_tei.begin(),m_tei.end(),0.0);

        // Compute all the relevant integrals
        for(size_t ix=0; ix < m_nx; ix++)
        for(size_t iy=0; iy < m_ny; iy++)
        for(size_t iz=0; iz < m_nz; iz++)
        {
            // Current site index
            size_t p, q;
            site_index(ix,iy,iz,p);
            // Set overlap matrix
            set_ovlp(p,p,1.0);
            // Set two-electron integrals
            set_tei(p,p,p,p,U);
            // Dipole terms (in order x,y,z)
            m_dipole[0*n2+p*m_nbsf+p] = ix-cx;
            m_dipole[1*n2+p*m_nbsf+p] = iy-cy;
            m_dipole[2*n2+p*m_nbsf+p] = iz-cz;
            // x-hops
            if(site_index(ix+1,iy,iz,q)) m_oei_a[p*m_nbsf+q] -= t;
            if(site_index(ix-1,iy,iz,q)) m_oei_a[p*m_nbsf+q] -= t;
            // y-hops
            if(site_index(ix,iy+1,iz,q)) m_oei_a[p*m_nbsf+q] -= t;
            if(site_index(ix,iy-1,iz,q)) m_oei_a[p*m_nbsf+q] -= t;
            // z-hops
            if(site_index(ix,iy,iz+1,q)) m_oei_a[p*m_nbsf+q] -= t;
            if(site_index(ix,iy,iz-1,q)) m_oei_a[p*m_nbsf+q] -= t;
        }
        // Copy to beta
        m_oei_b.resize(m_nbsf*m_nbsf);
        std::fill(m_oei_b.begin(),m_oei_b.end(),0.0);
        for(size_t pq=0; pq < m_nbsf * m_nbsf; pq++)
            m_oei_b[pq] = m_oei_a[pq];

        // Initialise orthogonalisation matrix
        compute_orthogonalization_matrix();
    }

private:
    int m_nx, m_ny, m_nz; ///!< Number of lattice sites in x, y, z directions
    double m_U = 0; ///!< Hubbard U parameter
    double m_t = 1; ///!< Hopping parameter 
 
    bool m_pX; ///!< Periodic boundary conditions in x
    bool m_pY; ///!< Periodic boundary conditions in y
    bool m_pZ; ///!< Periodic boundary conditions in z

    std::vector<double> m_tmat; ///!< Matrix of hopping connections

    /// @brief Get index of a given site
    /// @param x x-coordinate of the site
    /// @param y y-coordinate of the site
    /// @param z z-coordinate of the site
    bool site_index(int x, int y, int z, size_t &index) 
    {
        // Apply periodic boundary conditions in x direction
        if(x<0 and m_pX) x += m_nx;
        else if(x>=m_nx and m_pX) x -= m_nx;
        // Apply periodic boundary conditions in y direction
        if(y<0 and m_pY) y += m_ny;
        else if(y>=m_ny and m_pY) y -= m_ny;
        // Apply periodic boundary conditions in z direction
        if(z<0 and m_pZ) z += m_nz;
        else if(z>=m_nz and m_pZ) z -= m_nz;
        // Return false if any of the indices are out of bounds
        if(x<0 or x>=m_nx or y<0 or y>=m_ny or z<0 or z>=m_nz) return false;
        // Otherwise set index and return true
        index = x + y*m_nx + z*m_nx*m_ny;
        return true;
    }

    /// @brief Get site index of a given basis function as a tuple (ix,iy,iz)
    /// @param p basis function index
    std::tuple<int,int,int> site_index(size_t p)
    {
        int x = p % m_nx;
        int y = (p / m_nx) % m_ny;
        int z = p / (m_nx * m_ny);
        return std::make_tuple(x,y,z);
    }
};

#endif // HUBBARD_INTERFACE_H