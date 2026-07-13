#include "gUGA_ci_space.h" 
#include <armadillo> 
#include "omp_device.h"
#include <cstdint>
#include <omp.h>
#include "gUGA_evaluator.h"

void GUGA_CIspace::initialize(std::string citype, std::vector<std::string> configlist){ 
    // Check that we haven't already initialized CI space
    if(m_initialized)
        throw std::runtime_error("GUGA_CIspace::constructor CI space has already been initialised!");

    // Transform CI type to uppercase
    std::transform(citype.begin(), citype.end(), citype.begin(),
                   [](unsigned char c){ return std::toupper(c); });
                   
    if(citype == "FCI") 
        this->build_fci_configs();
    else 
        throw std::runtime_error("GUGA_CIspace::constructor CI type not implemented");

    
    // Check m_configs map 
    // Build memory maps
    build_memory_map1();
    build_memory_map2();
    //resolve_build_memory_map2();
   
    // Record that we successfully initialised
    m_initialized = true;
}

std::vector<Configuration> GUGA_CIspace::get_basis() {
    std::vector<Configuration> basis ; 
    for (auto& [ config, ind] : m_configs) { 
        basis.push_back(config); 
    } 
    return basis; 
}

void GUGA_CIspace::build_fci_configs() { 

    std::vector<Configuration> basis = this->m_drtobj.build_fci_configs() ; 
    m_nconfigs = 0 ;
    for (int a=0 ; a<(int) basis.size() ; a++){
        m_configs[basis[a]] = m_nconfigs++;
    }
}

void GUGA_CIspace::build_memory_map1() {
    // Populate m_map with connected configurations  
    //#pragma omp parallel for collapse(2)
    //std::cout << " building memory map 1 " << std::endl ; 
    for (size_t p=1 ; p<=m_nmo ; p++){ 
        for (size_t q=1 ; q<=m_nmo ; q++){ 
            // Iterate overall the diagonal operators and R operators 
            if(p>q) continue;
            //std::cout << "p,q " << std::to_string(p) << "," << std::to_string(q) << std::endl;  
            // Make an excitation 
            Eph Epq = {p,q}; 
            Eph Eqp = {q,p};
            
            // Initialise map vectors 
            #pragma omp critical 
            { 
                m_map1[Epq] = std::vector<std::tuple<size_t,size_t,double>> (); 
                if (Epq != Eqp) { 
                    m_map1[Eqp] = std::vector<std::tuple<size_t,size_t,double>> (); 
                }
            }
            
            // Loop over configurations 
            // Does this catch the case in which this is empty?
            for (auto &[configJ, indJ] : m_configs) { 
                // Need to loop over the possible excitations
                // Compute the R operator excitations 
                //std::cout << "testing 1 inside build" << std::endl; 
                std::vector<std::tuple<Configuration, double>> excitations = this->m_drtobj.apply_excitation(configJ,Epq); 
                for (auto ex : excitations) {
                    //std::cout << "testing 2 inside build" << std::endl; 
                    try {
                        //std::cout << "testing 2.1 inside build" << std::endl; 
                        Configuration bra = std::get<0> (ex) ; 
                        //std::cout << "ket " << configJ.config_str() << " bra " << bra.config_str() << std::endl ;  
                        //if (m_configs.find(std::get<0>(ex)) == m_configs.end()) {
                        //    std::cout << "bra not found in map!" << std::endl;
                        //} else {
                        //    std::cout << "bra found" << std::endl;
                        //}

                        size_t indI = m_configs.at(std::get<0> (ex)); 
                        //std::cout << "testing 2.2 inside build" << std::endl; 
                        m_map1[Epq].push_back(std::make_tuple(indJ,indI,std::get<1>(ex))) ; 
                        if (Eqp != Epq ) { 
                            // L operator relationships
                            m_map1[Eqp].push_back(std::make_tuple(indI,indJ,std::get<1>(ex))) ; 
                        }
                        //std::cout << "testing 3 inside build" << std::endl; 
                        std::vector<std::tuple<size_t,size_t,double>> excitations = m_map1.at(Epq);
                        //for (auto &[indJ, indI, element] : excitations) {
                        //    if (std::abs(element) > 1e-8) {
                        //        std::cout << "INSIDE MAP: indJ=" << indJ << " indI=" << indI << " element=" << element << std::endl;
                        //    }
                        //}
                    } 
                    catch( const std::out_of_range& e) { 
                        continue; 
                    }
                }
            }
        }
    }
}

void GUGA_CIspace::build_memory_map2(){ 
    // Populate m_map with connected determinants
    //#pragma omp parallel for collapse(4)
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t q=1; q<=m_nmo; q++)
    for(size_t r=1; r<=m_nmo; r++)
    for(size_t s=1; s<=m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;
        if(pq > rs) continue;
        Epphh Epqrs = {p,q,r,s};

        // Initialise map vectors
        m_map2[Epqrs] = std::vector<std::tuple<size_t,size_t,double> >();

        // Loop over determinants
        for(auto &[configJ, indJ] : m_configs)
        {
            { 
                std::vector<std::tuple<Configuration,double>> excitations = m_drtobj.apply_excitation(configJ, Epqrs);
                for (auto ex : excitations) {
                    try { 
                        size_t indI = m_configs.at(std::get<0> (ex));
                        // Does that mean we are double counting contributions? 
                        m_map2[Epqrs].push_back(std::make_tuple(indJ,indI,std::get<1>(ex))) ; 
                    } 
                    catch( const std::out_of_range& e) { 
                        continue; 
                    }
                }
            }
        }
    }
}

void GUGA_CIspace::resolve_build_memory_map2(){ 

    // Populate m_map with connected determinants
    //#pragma omp parallel for collapse(4)
    std::vector<Configuration> basis = m_drtobj.build_fci_configs() ;
    GUGAEval mb ;  
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t q=1; q<=m_nmo; q++)
    for(size_t r=1; r<=m_nmo; r++)
    for(size_t s=1; s<=m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;
        if(pq > rs) continue;
        Epphh Epqrs = {p,q,r,s};

        // Initialise map vectors
        m_map2[Epqrs] = std::vector<std::tuple<size_t,size_t,double> >();

        // Loop over determinants
        for(size_t bind = 0 ; bind < basis.size() ; bind++)
        {
            for(size_t kind = 0 ; kind < basis.size() ; kind++)
            {
                double matel = mb.two_body_coupling(basis[bind], basis[kind], Epqrs) ; 
                if ((std::abs(matel) != 0 ) && ( std::abs(matel) > 1e-5 ) ) {
                    int IndBra = this->get_config_index(basis[bind]); 
                    int IndKet = this->get_config_index(basis[kind]); 
                    m_map2[Epqrs].push_back(std::make_tuple(IndKet,IndBra,matel)) ; 
                } 
            }
        }
    }
}

void GUGA_CIspace::H_on_vec(const std::vector<double> &ci_vec, std::vector<double> &sigma) { 
    if(!m_initialized)
        throw std::runtime_error("GUGA_CIspace::H_on_vec: CI space has not been initialized!");

    // Check size of input
    if(ci_vec.size() != m_nconfigs) 
        throw std::runtime_error("GUGA_CIspace::H_on_vec: CI vector size error");

    // Get information about the OpenMP device
    omp_device dev;
    // Tolerance 
    double tol = m_ints.tol();
   
    std::vector<double> sigma_t(dev.nthreads*m_nconfigs);
    std::fill(sigma_t.begin(), sigma_t.end(), 0.0);
    
    // One-electron part
    //#pragma omp parallel for collapse(2)
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t q=1; q<=m_nmo; q++)
    {
        size_t ithread = dev.thread_id();
        double *st = &sigma_t[ithread*m_nconfigs];

        // Get the integral factos     
        double hpq = m_ints.oei(p-1,q-1);
        if(std::abs(hpq) > tol)
        {
            // Use excitation as a key to get all the connected determinants 
            for(auto &[indJ, indI, coeff] : m_map1.at({p,q}))
                st[indI] += coeff * hpq * ci_vec[indJ];
        }
    }

    // Two-electron part
    //#pragma omp parallel for collapse(4)
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t r=1; r<=m_nmo; r++)
    for(size_t q=1; q<=m_nmo; q++)
    for(size_t s=1; s<=m_nmo; s++)
    {
        // Consider only unique pairs
        // we need to limit the sums because we didnt add all the excitations into the map earlier 
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;
        if(pq > rs) continue;

        // Access memory 
        size_t ithread = dev.thread_id();
        double *st = &sigma_t[ithread*m_nconfigs];
        
        // Get two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>
        double vpqrs = 0.5*m_ints.tei(p-1,q-1,r-1,s-1);
        if(std::abs(vpqrs) > tol) {
            for(auto &[indJ, indI, coeff] : m_map2.at({p,q,r,s}))
            {
                st[indI] += coeff * vpqrs * ci_vec[indJ];
                if(pq!=rs) st[indJ] += coeff * vpqrs * ci_vec[indI];
            }
        }
    }

    // Initialise resulting sigma vector
    sigma.resize(m_nconfigs,0.0);
    double v_scalar = m_ints.scalar_potential();
    for(size_t ind=0; ind<m_nconfigs; ind++)
        sigma[ind] += ci_vec[ind] * v_scalar;

    // Compile results from all threads
    for(size_t ind=0; ind<m_nconfigs; ind++)
    for(size_t ithread=0; ithread<dev.nthreads; ithread++)
        sigma[ind] += sigma_t[ithread*m_nconfigs+ind];
} 

void GUGA_CIspace::build_Hd(std::vector<double> &Hdiag){ 
    // Get information about the OpenMP device
    omp_device dev;
    
    // Get thread-safe memory
    std::vector<double> Hdiag_t(dev.nthreads*m_nconfigs);
    std::fill(Hdiag_t.begin(), Hdiag_t.end(), 0.0);

    // Add one-electron part (only diagonals contribute)
    double tol = m_ints.tol();
    #pragma omp parallel for
    for(size_t p=1; p<=m_nmo; p++)
    {
        // Get thread memory buffer
        int ithread = dev.thread_id();
        double *Ht = &Hdiag_t[ithread*m_nconfigs];

        // Get one-electron alfa integral
        double hpp = m_ints.oei(p-1,p-1);
        for(auto &[indJ, indI, coeff] : m_map1.at({p,p})){
            if(indJ==indI) { 
                Ht[indI] += coeff * hpp;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t q=1; q<=m_nmo; q++)
    {
        if (p>q) continue ;  
        
        // Get thread memory buffer
        int ithread = dev.thread_id();
        double *Ht = &Hdiag_t[ithread*m_nconfigs];

        // Only include number-preserving terms
        if ( p == q ) { 
            double vpppp = 0.5 * (m_ints.tei(p-1,p-1,p-1,p-1)) ;
            if((std::abs(vpppp) > tol))
            {
                for(auto &[indJ, indI, coeff] : m_map2.at({p,p,p,p})) { 
                    if (indJ==indI) { 
                        Ht[indI] += coeff * vpppp;
                    }
                }
            }
        }
        else { 
            double vpqpq = 0.5 * m_ints.tei(p-1,q-1,p-1,q-1);
            if((std::abs(vpqpq) > tol))
            {
                for(auto &[indJ, indI, coeff] : m_map2.at({p,q,p,q})) { 
                    if (indI==indJ) { 
                        Ht[indI] += coeff * vpqpq;
                    }
                }
                for(auto &[indJ, indI, coeff] : m_map2.at({q,p,q,p})) { 
                    if (indI==indJ) { 
                        Ht[indI] += coeff * vpqpq;
                    }
                }
            }

            // Always select the correct pair
            // Already caught all p==q instances
            double vpqqp = 1 * m_ints.tei(p-1,q-1,q-1,p-1);
            if((std::abs(vpqqp) > tol)){
                for(auto &[indJ, indI, coeff] : m_map2.at({p,q,q,p})) {
                    if (indJ==indI) { 
                        Ht[indI] += coeff * vpqqp;
                    }
                }
            }
        }
    }

    // Collect final results
    Hdiag.resize(m_nconfigs);
    std::fill(Hdiag.begin(), Hdiag.end(), m_ints.scalar_potential());
    for(size_t it=0; it < dev.nthreads; it++)
    for(size_t ind=0; ind < m_nconfigs; ind++)
        Hdiag[ind] += Hdiag_t[it*m_nconfigs+ind];
}

void GUGA_CIspace::resolve_build_Hd(std::vector<double> &Hdiag){ 
    // Get information about the OpenMP device
    omp_device dev;
    
    // Get thread-safe memory
    std::vector<double> Hdiag_t(dev.nthreads*m_nconfigs);
    std::fill(Hdiag_t.begin(), Hdiag_t.end(), 0.0);

    // Add one-electron part (only diagonals contribute)
    double tol = m_ints.tol();
    #pragma omp parallel for
    for(size_t p=1; p<=m_nmo; p++)
    {
        // Get thread memory buffer
        int ithread = dev.thread_id();
        double *Ht = &Hdiag_t[ithread*m_nconfigs];

        // Get one-electron alfa integral
        double hpp = m_ints.oei(p-1,p-1);
        for(auto &[indJ, indI, coeff] : m_map1.at({p,p})){
            if(indJ==indI) { 
                Ht[indI] += coeff * hpp;
            }
        }
    }

    for (size_t p=1;p<=m_nmo;p++){ 
        for (size_t q=1;q<=m_nmo;q++) { 
            for (size_t r=1;r<=m_nmo;r++) { 
                for (size_t s=1;s<=m_nmo;s++) {
                    // Get thread memory buffer
                    Epphh Epqrs = {p,q,r,s} ;   
                    double vpqrs = 0.5*m_ints.tei(p-1,q-1,r-1,s-1);
                    for (auto [config, ind] : m_configs) { 
                        double matrix_element = this->resolve_two_body_matrix_element(config, config, Epqrs) ; 
                        Hdiag_t[ind] += matrix_element * vpqrs ; 
                    }
                }
            }
        }
    }

    // Collect final results
    Hdiag.resize(m_nconfigs);
    std::fill(Hdiag.begin(), Hdiag.end(), m_ints.scalar_potential());
    for(size_t it=0; it < dev.nthreads; it++)
    for(size_t ind=0; ind < m_nconfigs; ind++)
        Hdiag[ind] += Hdiag_t[it*m_nconfigs+ind];
}

double GUGA_CIspace::resolve_two_body_matrix_element( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs) const { 
    // Check same number of electrons, orbitals and S 
    if ( (bra.m_nmo != ket.m_nmo) || (bra.m_nelec != ket.m_nelec) || (bra.m_totspin != ket.m_totspin)  ) { 
        std::cout << "Different N, n or S values" << std::endl ; 
        return 0.0 ; 
    }
    const size_t nmo = bra.m_nmo ;
    const arma::imat bra_paldus = bra.generate_paldus() ; 
    const arma::imat ket_paldus = ket.generate_paldus() ;

    // Make sure both sets of indices are in range 
    size_t i = Epqrs.particle1 ; 
    size_t j = Epqrs.hole1 ; 
    size_t k = Epqrs.particle2 ; 
    size_t l = Epqrs.hole2 ;
    size_t head = std::max({i,j,k,l}) ;  
    size_t tail = std::min({i,j,k,l}) ;  
    assert( head <= bra.m_nmo ) ;
    assert( tail > 0 ) ; 
    
    // Check path outside loop
    // dont need to check a=0 as that is always 0.  
    for (size_t a = 1 ; a <= nmo ; a++ ) {
        // iterate over the full loop  
        if ( ( a >= head ) || ( a < tail )  ) {
            // if a is outside the "true_loop" 
            if (!arma::all( bra_paldus.row(a) == ket_paldus.row(a) ) ) { 
                return 0.0 ; 
            }
        } 
    }

    double matrix_element = 0 ; 
    GUGAEval mb ;  
    for (auto [ config, IndI ] : m_configs) {
        matrix_element += mb.one_body_coupling(bra, config, {i,j})*mb.one_body_coupling(config, ket, {k,l}) ;   
    }
    if (j==k) { 
        matrix_element -= mb.one_body_coupling(bra, ket,{i,l}) ;  
    }
    return matrix_element ; 
}

// Functions to construct Hamiltonian matrices 
void GUGA_CIspace::build_Hmat(std::vector<double> &Hmat)
{
    if(!m_initialized)
        throw std::runtime_error("GUGA_CIspace::build_Hmat: space has not been initialized!");

    // Check size of output and initialise memory
    Hmat.resize(m_nconfigs*m_nconfigs);
    std::fill(Hmat.begin(), Hmat.end(), 0.0);

    // Scalar component
    build_H0(Hmat);
    // One-electron component
    build_H1(Hmat);
    // Two-electron component
    build_H2(Hmat);
} 

void GUGA_CIspace::build_H0(std::vector<double> &H0)
{
    // Diagonal scalar part
    double v_scalar = m_ints.scalar_potential();
    //#pragma omp parallel for
    for(size_t I=0; I<m_nconfigs; I++)
        H0[I*m_nconfigs+I] += v_scalar;
}

void GUGA_CIspace::build_H1(std::vector<double> &H1)
{
    // Get relevant memory map
    double tol = m_ints.tol();

    // Get one-electron integrals
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t q=p; q<=m_nmo; q++)
    {
        double hpq = m_ints.oei(p-1,q-1);
        if(std::abs(hpq) < tol) continue;

        for(auto &[indJ, indI, coeff] : m_map1.at({p,q}))
        {
            H1[indI*m_nconfigs+indJ] += coeff * hpq;
            if(p!=q) 
                H1[indJ*m_nconfigs+indI] += coeff * hpq;
        }
    }
}

void GUGA_CIspace::build_H2(std::vector<double> &H2)
{
    double tol = m_ints.tol();
    // Get relevant memory map
    for(size_t p=1; p<=m_nmo; p++)
    for(size_t r=1; r<=m_nmo; r++)
    for(size_t q=1; q<=m_nmo; q++)
    for(size_t s=1; s<=m_nmo; s++)
    {
        // Consider only unique pairs
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;
        if(pq > rs) continue;

        double vpqrs = 0.5*m_ints.tei(p-1,q-1,r-1,s-1) ;
        double vrspq = 0.5*m_ints.tei(r-1,s-1,p-1,q-1) ;

        if((std::abs(vpqrs) < tol) and (std::abs(vrspq) < tol)) 
            continue;

        for(auto &[indJ, indI, coeff] : m_map2.at({p,q,r,s}))
        {
            H2[indI*m_nconfigs+indJ] += coeff * vpqrs;
            if(pq!=rs) H2[indJ*m_nconfigs+indI] += coeff * vrspq;
        }
    }
}

// Helper functions
/// Print the configuration list
void GUGA_CIspace::print() const {
    if(!m_initialized)
        throw std::runtime_error("GUGA_CIspace::print: space has not been initialized!");

    for(auto &[config, index] : m_configs)
        std::cout << config.config_str()  << ": " << index << std::endl;
}
/// Print a CI vector
void GUGA_CIspace::print_vector(const std::vector<double> &ci_vec, double tol) const{
    if(!m_initialized)
        throw std::runtime_error("GUGA_CIspace::print_vector: space has not been initialized!");

    if(ci_vec.size() != m_nconfigs) 
        throw std::runtime_error("GUGA_CIspace::print_vector: CI vector size error");
    
    for(auto &[config, ind] : m_configs)
    {
        if(std::abs(ci_vec[ind]) > tol) 
            fmt::print("{:>s}: {:>10.6f}\n", config.config_str() , ci_vec[ind]);;
    }   
}

void GUGA_CIspace::print_map_couplings(Configuration &bra, Configuration &ket) {
    size_t ind_bra = (size_t) this->get_config_index(bra);
    size_t ind_ket = (size_t) this->get_config_index(ket);
    std::cout << "Map couplings between bra: " << bra.config_str() << " (ind=" << ind_bra << ")"
              << " and ket: " << ket.config_str() << " (ind=" << ind_ket << ")" << std::endl;

    // One-body operators
    std::cout << "\n--- One-body operators (via m_map1) ---" << std::endl;
    for (size_t p = 1; p <= m_nmo; p++)
    for (size_t q = 1; q <= m_nmo; q++) {
        try {
            for (auto &[indJ, indI, coeff] : m_map1.at({p,q})) {
                if (indJ == ind_ket && indI == ind_bra) {
                    std::cout << "  E(" << p << "," << q << ") coeff=" << coeff << std::endl;
                }
            }
        } catch (const std::out_of_range &e) { continue; }
    }

    // Two-body operators
    std::cout << "\n--- Two-body operators (via m_map2) ---" << std::endl;
    for (size_t p = 1; p <= m_nmo; p++)
    for (size_t q = 1; q <= m_nmo; q++)
    for (size_t r = 1; r <= m_nmo; r++)
    for (size_t s = 1; s <= m_nmo; s++) {
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;

        if (pq > rs ) continue ; 
        Epphh Epqrs = {p,q,r,s} ;

        for (auto &[indJ, indI, coeff] : m_map2.at(Epqrs)) {
            if (indJ == ind_ket && indI == ind_bra) {
                std::cout << "  e(" << p << "," << q << "," << r << "," << s
                          << ") coeff=" << coeff << std::endl;
            }
        }
    }
}

std::vector<std::tuple<Eph,double>> GUGA_CIspace::map1_couplings(Configuration &bra, Configuration &ket) {
    size_t ind_bra = (size_t) this->get_config_index(bra);
    size_t ind_ket = (size_t) this->get_config_index(ket);

    std::vector<std::tuple<Eph,double>> result ; 
    // One-body operators
    for (size_t p = 1; p <= m_nmo; p++)
    for (size_t q = 1; q <= m_nmo; q++) {
        try {
            for (auto &[indJ, indI, coeff] : m_map1.at({p,q})) {
                if (indJ == ind_ket && indI == ind_bra) {
                    Eph Epq = {p,q }; 
                    result.push_back(std::make_tuple(Epq, coeff));
                }
            }
        } catch (const std::out_of_range &e) { continue; }
    }
    return result; 
}

std::vector<std::tuple<Epphh,double>> GUGA_CIspace::map2_couplings(Configuration &bra, Configuration &ket) {
    size_t ind_bra = (size_t) this->get_config_index(bra);
    size_t ind_ket = (size_t) this->get_config_index(ket);

    std::vector<std::tuple<Epphh,double>> result ; 
    // Two-body operators
    for (size_t p = 1; p <= m_nmo; p++)
    for (size_t q = 1; q <= m_nmo; q++)
    for (size_t r = 1; r <= m_nmo; r++)
    for (size_t s = 1; s <= m_nmo; s++) {
        size_t pq = p*(m_nmo+1) + q;
        size_t rs = r*(m_nmo+1) + s;

        if (pq > rs ) continue ; 
        Epphh Epqrs = {p,q,r,s} ;
        for (auto &[indJ, indI, coeff] : m_map2.at(Epqrs)) {
            if (indJ == ind_ket && indI == ind_bra) {
                    result.push_back(std::make_tuple(Epqrs, coeff));
            }
        }
    }
    return result; 
}
