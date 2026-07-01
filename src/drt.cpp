#include <string>
#include <algorithm>
#include <vector> 
#include <armadillo>
#include "drt.h"
#include "configuration.h"
#include "excitation.h"
#include "matrix_element_calculator.h"
#include <set>

void DRT::construct_drt() { 
    // Make vector for dynamic sizing 
    std::vector<std::vector<int>> soln_array ;
    // Top row of drt, with level ind, connecting d0 d1 d2 d3 (leixcal ind)
    
    int nmo = m_a + m_b + m_c ; 
    soln_array.push_back( { m_a, m_b, m_c, nmo, 0, 0, 0, 0} ) ; 
    // Connected solns is remade? 
    std::vector<size_t> connected_solns = { 0 };  
    // Iterate through level index from nmo to 0 
    for (int i = nmo ; i >= 0 ; i--) { 
        // start number of vector elements in soln_array 
        // so start is the lexical index - 1 ?  
        size_t start = soln_array.size() ;
        std::vector<std::vector<int>> temp_solns; 
        std::vector<size_t> temp_inds ; 
        // so iterate over number of connected solns? 
        for (size_t j = 0 ; j < connected_solns.size() ; j++) {
            // starting row of drt 
            size_t prev = connected_solns[j] ; 
            // iterate through possible steps 
            for (uint8_t d = 0 ; d < 4 ; d++) { 
                // Connect by one of the step vectors
                std::vector<int> current(8);
                // save the a,b,c value
                current[0] = soln_array[prev][0] - step_vecs(d, 0);
                current[1] = soln_array[prev][1] - step_vecs(d, 1);
                current[2] = soln_array[prev][2] - step_vecs(d, 2);
                // save the level index 
                current[3] =  i-1;
                // Save solution name rather than index
                //  0 == "None" stand-in
                // saves lexical index of the top solution this row couples to via 
                // whichever step vector   
                current[4+d] = prev + 1 ; 
                // this enforces all these terms to be positive 
                if ( (current[0]>=0) && (current[1]>=0) && (current[2]>=0) ) { 
                    bool inlist = false; 
                    // check that we havent already located this solution
                    // if we have then we just update the corresponding di step element with 
                    // other connecting top row 
                    for (size_t k = 0; k < temp_solns.size(); k++) { 
                        if ( (temp_solns[k][0] == current[0]) && (temp_solns[k][1] == current[1]) && (temp_solns[k][2] == current[2]) ) { 
                            temp_solns[k][4+d] = prev + 1; 
                            inlist = true ; 
                            break; 
                        }
                    }
                    if (!inlist) { 
                        temp_solns.push_back(current);
                        temp_inds.push_back(start); 
                        start++; 

                    }
                }
            }
        }
        for (const auto& sol : temp_solns) { 
            soln_array.push_back(sol);
        }
        // now define the connected_solns as all the current level index solutions
        // ready for the next iteration 
        connected_solns = temp_inds ; 
    }

    // finally we translate the vector of vectors into an arma array 
    arma::imat distinct_row_table(soln_array.size(), 8);
    for (size_t i = 0; i < soln_array.size(); i++) {
        for (size_t j = 0; j < 8; j++) {
            distinct_row_table(i, j) = soln_array[i][j];
        }
    }
    
    // So this more than just the DRT, it also tracks all the connectivity between solutions
    m_drt = distinct_row_table; 
    return ; 
}

arma::uvec DRT::one_body_step(int &ref_step, int &level, Eph &Epq) { 
    
    int p = (int) Epq.particle; 
    int q = (int) Epq.hole; 

    std::string oclass ; 
    std::vector<int> allowed_steps  = {} ; 
    //
    if ( p < q ) { oclass = "R" ; } 
    else if ( p > q ) { oclass = "L" ; }
    else { 
        allowed_steps = {ref_step} ;
        arma::uvec final = arma::conv_to<arma::uvec>::from(allowed_steps);  
        return final;  
    } 
    
    if (level==std::min(p,q)) { oclass = "t" + oclass ;}
    else if (level==std::max(p,q)) { oclass = "h" + oclass ;}

    if ( (oclass=="tR") || (oclass=="hL") ) { 
        if (ref_step==0) { allowed_steps = {1,2} ;}
        else if ( (ref_step==1) || (ref_step==2) ) { allowed_steps = {0} ;}
    }
    else if ( (oclass=="hR") || (oclass=="tL") ) { 
        if ( ref_step==3) { allowed_steps={1,2} ;}
        else if ( (ref_step==1) || (ref_step==2) ) { allowed_steps = {0} ; }
    } 
    else if ( (oclass=="R") || (oclass=="L") ) {
        if ( (ref_step==3) || (ref_step==0)) { allowed_steps={ref_step} ;}
        else if ( (ref_step==1) || (ref_step==2) ) { allowed_steps = {1,2} ; }
    } 

    arma::uvec final = arma::conv_to<arma::uvec>::from(allowed_steps); 
    return final;  
} 

arma::uvec DRT::two_body_step(int &ref_step, int &level, Epphh &Epqrs) { 
    int i = (int) Epqrs.particle1; 
    int j = (int) Epqrs.hole1; 
    int k = (int) Epqrs.particle2; 
    int l = (int) Epqrs.hole2; 

    // Need a check that the level is inside the range of the excitation operator 
    // if ((level > head) || (level < tail )) { yieldError }
    if (level > std::max({i,j,k,l}) || (level < std::min({i,j,k,l})) ) { 
        std::cerr << "DRT::two_body_step error : level is outside loop range" << std::endl ; 
    }

    std::vector<int> allowed_steps ; 
    arma::uvec final ; 
    std::vector<std::string> op_classes ; 

    // Catch what are actually one body contributions  
    // S2 contributions 
    if ( ( (level <= std::max(i,j)) && (level >= std::min(i,j)) ) 
        && (not ( (level <= std::max(k,l)) && (level >= std::min(k,l)) ) ) ){ 
        // level in (i,j) but not in (k,l)
        Eph Eij={(size_t)i,(size_t)j}; 
        final = this->one_body_step(ref_step, level, Eij) ; 
        return final ; 
    } 
    else if ( ( not( (level <= std::max(i,j)) && (level >= std::min(i,j)) )            
        &&  ( (level <= std::max(k,l)) && (level >= std::min(k,l)) ) ) ) { 
        // level in (k,l) not in (i,j)
        Eph Ekl={(size_t) k, (size_t) l}; 
        final = this->one_body_step(ref_step, level, Ekl) ;      
        return final ; 
    }

    // Classify two body operators 
    if (i < j ) { op_classes.push_back("R"); }
    else if ( i > j ) { op_classes.push_back("L"); }
    else { std::cerr  << "Error: two_body_drt_step does not deal with diagonal value, should've been caught earlier" << std::endl;}
    //
    if (k < l ) { op_classes.push_back("R"); }
    else if ( k > l ) { op_classes.push_back("L"); }
    else { std::cerr  << "Error: two_body_drt_step does not deal with diagonal value, should've been caught earlier" << std::endl;}
    // 
    std::vector<int> heads = { std::max(i,j), std::max(k,l)}; 
    std::vector<int> tails = { std::min(i,j), std::min(k,l)}; 
    //
    if (level==heads[0]) { op_classes[0] = "h" + op_classes[0] ; }
    if (level==heads[1]) { op_classes[1] = "h" + op_classes[1] ; }
    if (level==tails[0]) { op_classes[0] = "t" + op_classes[0] ; }
    if (level==tails[1]) { op_classes[1] = "t" + op_classes[1] ; }

    // Helper function to check if value is in vector
    auto contains = [&](const std::string& val) {
        return std::find(op_classes.begin(), op_classes.end(), val) != op_classes.end();
    };

    if ( (op_classes == std::vector<std::string>{"hR","hR"})
      || (op_classes == std::vector<std::string>{"tL","tL"})
      || (contains("hR") && contains("tL")) ) {
        if (ref_step == 3) {
            allowed_steps = {0};
        }
    } else if ( (op_classes == std::vector<std::string>{"tR","tR"})
              || (op_classes == std::vector<std::string>{"hL","hL"})
              || (contains("tR") && contains("hL")) ) {
        if (ref_step == 0) {
            allowed_steps = {3};
        }
    } else if ( (contains("hR") && contains("hL"))
              || (contains("tR") && contains("tL"))
              || (contains("tR") && contains("hR"))
              || (contains("tL") && contains("hL"))
              || (op_classes == std::vector<std::string>{"R","R"})
              || (op_classes == std::vector<std::string>{"L","L"})
              || (contains("R") && contains("L")) ) {
        if (ref_step == 0 || ref_step == 3) {
            allowed_steps = {ref_step};
        } else if (ref_step == 1 || ref_step == 2) {
            allowed_steps = {1, 2};
        }
    } else if ( (contains("R") && contains("hR"))
              || (contains("L") && contains("tL"))
              || (contains("hR") && contains("L"))
              || (contains("R") && contains("tL")) ) {
        if (ref_step == 1 || ref_step == 2) {
            allowed_steps = {0};
        } else if (ref_step == 3) {
            allowed_steps = {1, 2};
        }
    } else if ( (contains("L") && contains("hL"))
              || (contains("R") && contains("tR"))
              || (contains("R") && contains("hL"))
              || (contains("tR") && contains("L")) ) {
        if (ref_step == 1 || ref_step == 2) {
            allowed_steps = {3};
        } else if (ref_step == 0) {
            allowed_steps = {1, 2};
        }
    } else {
        std::cerr << "Fell through Error: " << op_classes[0] << " " << op_classes[1] << std::endl;
    }
 
    final = arma::conv_to<arma::uvec>::from(allowed_steps); 
    return final;  
} 

std::vector<Configuration> DRT::drt_loop(Configuration &ref_config, int &head, int &tail, std::function<arma::uvec(int&,int&)> stepfunc) {
    std::vector<uint8_t> ket_vec = ref_config.get_vec() ;
    arma::irowvec start = {0,0,0} ; 
    arma::irowvec end = {0,0,0} ; 
    size_t start_ind = 0 ;
    size_t end_ind = 0 ;
    for (int i = 0 ; i < tail-1 ; i ++) {
        start += step_vecs.row(ket_vec[i]);
    }
    for (int i = 0 ; i < head ; i ++) {
        end += step_vecs.row(ket_vec[i]);
    }
    bool sbool  = false; 
    bool ebool = false; 
    for (size_t i = 0 ; i < m_drt.n_rows ; i ++ ) {
        if (!sbool && arma::all(m_drt.cols(0,2).row(i) == start)) {
            start_ind = i ;
            sbool = true;
        }
        if (!ebool && arma::all(m_drt.cols(0,2).row(i) == end)) {
            end_ind = i ;
            ebool = true;
        }
        if (sbool && ebool) { 
            break; 
        }
    }
    // matrix element calculator are going to set a different function, now the configuration we dont need to specify
    // Store step vector and node lexical name  
    std::vector< std::tuple< std::vector<int>, int> > configurations  = {std::make_tuple(std::vector<int> {}, start_ind+1)}; 
    // Loop body
    for ( int level = m_drt(start_ind,3) ; level < m_drt(end_ind,3) ; level++ ) { 
        int ref_step = ket_vec[level-1] ;
        
        // excitation doesn't change 
        arma::uvec allowed_steps = stepfunc(ref_step, level); 
        
        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
        std::vector<int> lexical_inds ; 
        // Extract current node positions 
        for ( auto conf : configurations ) { 
            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
                lexical_inds.push_back(std::get<1>(conf)-1); 
            }
        }

        // Iterate over all current node positions 
        for (auto index : lexical_inds){ 
            std::vector<int> steps ;
            arma::uvec temp_steps ; 
            if (level== head -1 ) { 
                temp_steps = arma::find(m_drt.row(index).cols(4, m_drt.n_cols-1) == end_ind+1);
            }
            else { 
                temp_steps = arma::find(m_drt.row(index).cols(4, m_drt.n_cols-1) != 0);
            }
            
            // See if these are allowed steps 
            for (auto s : temp_steps ) { 
                if (arma::any(allowed_steps == s)) { 
                    steps.push_back(s); 
                }
            }
            
            //
            if (steps.size() != 0 ) { 
                for (auto configuration : configurations) { 
                    if ( std::get<1>(configuration)==index+1) {
                        for (auto step : steps ){ 
                            std::vector<int> comb = std::get<0>(configuration) ;
                            comb.push_back( step );
                            new_configurations.push_back( std::make_tuple(comb, m_drt(index,4+step)));
                        }
                    }
                }
            }
        }
        configurations = new_configurations ; 
    }

    std::vector<Configuration> excitations ; 
    for (auto conf : configurations ) {
        if ( std::get<1>(conf) == end_ind + 1) { 
            std::vector comb( ket_vec.begin(), ket_vec.begin() + tail - 1) ;  
            comb.insert(comb.end(), std::get<0>(conf).begin(), std::get<0>(conf).end()) ;  
            comb.insert(comb.end(), ket_vec.begin() + head , ket_vec.end()) ;
            Configuration bra(comb) ;
            excitations.push_back(bra) ; 
        }    
    }
    //Sort greatest first
    std::sort(excitations.begin(), excitations.end(), std::less<Configuration>());
    return excitations ; 
}

std::vector<std::tuple<Configuration,double>> DRT::apply_excitation( Configuration &ket, Eph &Epq) {
    // Extract the hole and particle excitation indices
    int p = (int) Epq.particle ; 
    int q = (int) Epq.hole ;
    int head = std::max(p,q) ; 
    int tail = std::min(p,q) ;

    std::vector<Configuration> ex_configs ;  
    MatrixElementCalculator mec;
   
    // Catch Diagonal case 
    if ( p == q ) { 
        ex_configs.push_back(ket) ; 
    }
    else { 
        ex_configs = this->drt_loop(ket, head, tail
                    , [&](int &ref_step,int &level){ return this->one_body_step(ref_step, level, Epq );}
                    ) ;               
    }

    std::vector<std::tuple<Configuration,double>> excitations ;  
    for (Configuration bra : ex_configs ) {
        // Matrix Element Calculator 
        double factor = mec.one_body_coupling(bra, ket , Epq) ; 
        if (std::abs(factor) > 1e-8 ) { 
            excitations.push_back( std::make_tuple(bra, factor) ); 
        }
    }    
    return excitations ; 
}

std::vector<std::tuple<Configuration,double>> DRT::apply_excitation( Configuration &ket, Epphh &Epqrs) {
    // Extract the hole and particle excitation indices
    int i = (int) Epqrs.particle1 ; 
    int j = (int) Epqrs.hole1 ;
    int k = (int) Epqrs.particle2 ; 
    int l = (int) Epqrs.hole2 ;
    
    std::vector<Configuration> ex_configs ;  
    MatrixElementCalculator mec;
    // 
    int head = std::max({i,j,k,l}) ; 
    int tail = std::min({i,j,k,l}) ; 

    // Catch Diagonal cases 
    if ( (i==j) || (k==l) ) { 
        std::vector<Configuration> ex_configs ; 
        if ( (i==j) && (k==l)) { 
            ex_configs.push_back( ket ) ;
        }
        else if ( (i==j) ) { 
            Eph Ekl = {(size_t)k,(size_t)l} ; 
            ex_configs = this->drt_loop(ket, head, tail
                    , [&](int &ref_step, int &level){ return this->one_body_step(ref_step, level, Ekl );}
                    ) ;
        }
        else if ( (k==l) ) { 
            Eph Eij = {(size_t)i,(size_t)j} ; 
            ex_configs = this->drt_loop(ket, head, tail
                    , [&](int &ref_step, int &level){ return this->one_body_step(ref_step, level, Eij );}
                    ) ;
        }

        if (j==k) { 
            std::vector<Configuration> temp_configs ; 
            Eph Eil = {(size_t)i,(size_t)l} ; 
            temp_configs = this->drt_loop(ket, head, tail
                    , [&](int &ref_step, int &level){ return this->one_body_step(ref_step, level, Eil );}
                    ) ;
            std::set<Configuration> tempset(ex_configs.begin(), ex_configs.end() ) ;  
            for (Configuration temp : temp_configs) { 
                if (tempset.count(temp)>0) { 
                    ex_configs.push_back(temp) ; 
                } 
            }
        }
    }
    else{ 

        ex_configs = this->drt_loop(ket, head, tail, [&]( int &ref_step, int &level){ return this->two_body_step(ref_step, level, Epqrs) ; } ) ; 
    }
    

    std::vector<std::tuple<Configuration,double>> excitations ;  
    for (Configuration bra : ex_configs ) {
        // Matrix Element Calculator 
        double factor = mec.two_body_coupling(bra, ket, Epqrs) ; 
        if (std::abs(factor) > 1e-8 ) { 
            excitations.push_back( std::make_tuple(bra, factor) ); 
        }
    }    
    return excitations ; 
}


