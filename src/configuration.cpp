#include "configuration.h"
#include "excitation.h" 
#include "matrix_element_calculator.h"
#include <armadillo> 
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

arma::imat Configuration::generate_paldus() const {
    // Implementation for generating Paldus table representation
    arma::imat paldus_table(m_nmo+1, 3, arma::fill::zeros);
    for (int i = 0; i < m_nmo ; i++){ 
        paldus_table.row(i+1) = paldus_table.row(i) + step_vecs.row(m_step_vec[i]); 
    }
    return paldus_table;
}

//std::vector<std::tuple<Configuration,double>> Configuration::old_apply_excitation(arma::imat &drt, const Eph &Epq) {
//    // Extract the hole and particle excitation indices
//    int p = (int) Epq.particle ; 
//    int q = (int) Epq.hole ;
//    assert(p<=q && "Function only locates D or R operator excitations");
//    std::vector<std::tuple<Configuration,double>> excitations ;  
//    MatrixElementCalculator mec;
//    
//    // Diagonal case 
//    if ( p == q ) { 
//        double factor = mec.one_body_coupling(*this, *this ,Epq) ; 
//        if (std::abs(factor) > 1e-8 ) { 
//            excitations.push_back( std::make_tuple(*this, factor) ); 
//        }
//        return excitations ; 
//    }
//    
//    // R operator case 
//    // Find all the solutions that have the same outisde levels
//    std::vector<uint8_t> ket_vec = this->get_vec() ;
//    arma::irowvec start = {0,0,0} ; 
//    arma::irowvec end = {0,0,0} ; 
//    size_t start_ind = 0 ;
//    size_t end_ind = 0 ;
//    for (int i = 0 ; i < p-1 ; i ++) {
//        start += step_vecs.row(ket_vec[i]);
//    }
//    for (int i = 0 ; i < q ; i ++) {
//        end += step_vecs.row(ket_vec[i]);
//    }
//    bool sbool  = false; 
//    bool ebool = false; 
//    for (size_t i = 0 ; i < drt.n_rows ; i ++ ) {
//        if (!sbool && arma::all(drt.cols(0,2).row(i) == start)) {
//            start_ind = i ;
//            sbool = true;
//        }
//        if (!ebool && arma::all(drt.cols(0,2).row(i) == end)) {
//            end_ind = i ;
//            ebool = true;
//        }
//        if (sbool && ebool) { 
//            break; 
//        }
//    }
//    
//    // Tail step 
//    arma::uvec steps ;
//    // Store step vector and node lexical name  
//    std::vector< std::tuple< std::vector<int>, int> > configurations ; 
//    // Scope declaration so level, ref_step and temp_steps isn't retained  
//    {
//        int ref_step = ket_vec[drt(start_ind,3)] ; 
//        arma::uvec temp_steps = arma::find(drt.row(start_ind).cols(4, drt.n_cols-1) != 0);
//        if (ref_step == 0 ) { 
//            steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps == 2))); 
//        }
//        else if (ref_step==1 || ref_step==2){
//           steps = temp_steps(arma::find(temp_steps == 3)); 
//        } 
//        else { 
//            return {}; 
//        }
//    }
//    for ( int step : steps ) { 
//        configurations.push_back( std::make_tuple(std::vector<int>{step}, drt(start_ind, 4 + step)  )) ; 
//    }
//    
//    // Loop body
//    for ( int level = drt(start_ind,3)+1 ; level < drt(end_ind,3) - 1 ; level++ ) { 
//        int ref_step = ket_vec[level] ;
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        // Extract current node positions 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        // Iterate over all current node positions 
//        for (auto index : lexical_inds){ 
//            steps.reset(); 
//            arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) != 0);
//            if (ref_step==0 || ref_step==3) { 
//                steps = temp_steps(arma::find(temp_steps == ref_step)); 
//            }
//            else if (ref_step==1 || ref_step==2) { 
//                steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps==2))); 
//            }
//            //
//            if (steps.is_empty()) { 
//                continue; 
//            }
//            //
//            for (auto configuration : configurations) { 
//                if ( std::get<1>(configuration)==index+1) {
//                    for (int step : steps) {
//                        std::vector<int> comb = std::get<0>(configuration) ;
//                        comb.push_back( step );
//                        new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//        // print configurations
//        //std::cout << "configurations: " << std::endl;
//        //for (auto conf : configurations) {
//        //    std::cout << "  steps: ";
//        //    for (auto s : std::get<0>(conf)) {
//        //        std::cout << s << " ";
//        //        }
//        //    std::cout << "  lexical name: " << std::get<1>(conf) << std::endl;
//        //}
//    }
//
//    // Head condition 
//    {
//        int ref_step = ket_vec[drt(end_ind,3)-2] ; 
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        for (auto index : lexical_inds){
//            steps.reset() ; 
//            arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) == end_ind + 1);
//            if (ref_step==1 || ref_step==2) { 
//                steps = temp_steps(arma::find(temp_steps == 0)); 
//            }
//            else if (ref_step==3) { 
//                steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps==2))); 
//            }
//            //
//            if (steps.is_empty()) { 
//                continue; 
//            }
//            //
//            for (auto configuration : configurations) { 
//                if ( std::get<1>(configuration)==index+1) {
//                    for (int step : steps) {
//                        std::vector<int> comb = std::get<0>(configuration) ;
//                        comb.push_back( step );
//                        new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//    }
//
//    for (auto conf : configurations ) {
//        if ( std::get<1>(conf) == end_ind + 1) { 
//            std::vector comb( ket_vec.begin(), ket_vec.begin() + p - 1) ;  
//            comb.insert(comb.end(), std::get<0>(conf).begin(), std::get<0>(conf).end()) ;  
//            comb.insert(comb.end(), ket_vec.begin() + q , ket_vec.end()) ;
//            Configuration bra(comb) ; 
//            double factor = mec.one_body_coupling(bra, *this ,Epq) ; 
//            if (std::abs(factor) > 1e-8 ) { 
//                excitations.push_back( std::make_tuple(bra, factor) ); 
//            }
//        }    
//    }
//    // Sort greatest first
//    //std::sort(final.begin(), final.end(), std::less<Configuration>());
//    return excitations ; 
//}
//
//// Now how can we do this effectively? 
//std::vector<std::tuple<Configuration,double>> Configuration::non_drt_apply_excitation(arma::imat &drt, const Eph &Epq) {
//    // Extract the hole and particle excitation indices
//    int p = (int) Epq.particle ; 
//    int q = (int) Epq.hole ;
//    assert(p<=q && "Function only locates D or R operator excitations");
//    std::vector<std::tuple<Configuration,double>> excitations ;  
//    MatrixElementCalculator mec;
//    
//    // Diagonal case 
//    if ( p == q ) { 
//        double factor = mec.one_body_coupling(*this, *this ,Epq) ; 
//        if (std::abs(factor) > 1e-8 ) { 
//            excitations.push_back( std::make_tuple(*this, factor) ); 
//        }
//        return excitations ; 
//    }
//    
//    // R operator case 
//    // Find all the solutions that have the same outisde levels
//    std::vector<uint8_t> ket_vec = this->get_vec() ;
//    arma::irowvec start = {0,0,0} ; 
//    arma::irowvec end = {0,0,0} ; 
//    size_t start_ind = 0 ;
//    size_t end_ind = 0 ;
//    for (int i = 0 ; i < p-1 ; i ++) {
//        start += step_vecs.row(ket_vec[i]);
//    }
//    for (int i = 0 ; i < q ; i ++) {
//        end += step_vecs.row(ket_vec[i]);
//    }
//    bool sbool  = false; 
//    bool ebool = false; 
//    for (size_t i = 0 ; i < drt.n_rows ; i ++ ) {
//        if (!sbool && arma::all(drt.cols(0,2).row(i) == start)) {
//            start_ind = i ;
//            sbool = true;
//        }
//        if (!ebool && arma::all(drt.cols(0,2).row(i) == end)) {
//            end_ind = i ;
//            ebool = true;
//        }
//        if (sbool && ebool) { 
//            break; 
//        }
//    }
//    
//    // Tail step 
//    arma::uvec steps ;
//    // Store step vector and node lexical name  
//    std::vector< std::tuple< std::vector<int>, int> > configurations ; 
//    // Scope declaration so level, ref_step and temp_steps isn't retained  
//    {
//        int ref_step = ket_vec[drt(start_ind,3)] ; 
//        arma::uvec temp_steps = arma::find(drt.row(start_ind).cols(4, drt.n_cols-1) != 0);
//        if (ref_step == 0 ) { 
//            steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps == 2))); 
//        }
//        else if (ref_step==1 || ref_step==2){
//           steps = temp_steps(arma::find(temp_steps == 3)); 
//        } 
//        else { 
//            return {}; 
//        }
//    }
//    for ( int step : steps ) { 
//        configurations.push_back( std::make_tuple(std::vector<int>{step}, drt(start_ind, 4 + step)  )) ; 
//    }
//    
//    // Loop body
//    for ( int level = drt(start_ind,3)+1 ; level < drt(end_ind,3) - 1 ; level++ ) { 
//        int ref_step = ket_vec[level] ;
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        // Extract current node positions 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        // Iterate over all current node positions 
//        for (auto index : lexical_inds){ 
//            steps.reset(); 
//            arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) != 0);
//            if (ref_step==0 || ref_step==3) { 
//                steps = temp_steps(arma::find(temp_steps == ref_step)); 
//            }
//            else if (ref_step==1 || ref_step==2) { 
//                steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps==2))); 
//            }
//            //
//            if (steps.is_empty()) { 
//                continue; 
//            }
//            //
//            for (auto configuration : configurations) { 
//                if ( std::get<1>(configuration)==index+1) {
//                    for (int step : steps) {
//                        std::vector<int> comb = std::get<0>(configuration) ;
//                        comb.push_back( step );
//                        new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//        // print configurations
//        //std::cout << "configurations: " << std::endl;
//        //for (auto conf : configurations) {
//        //    std::cout << "  steps: ";
//        //    for (auto s : std::get<0>(conf)) {
//        //        std::cout << s << " ";
//        //        }
//        //    std::cout << "  lexical name: " << std::get<1>(conf) << std::endl;
//        //}
//    }
//
//    // Head condition 
//    {
//        int ref_step = ket_vec[drt(end_ind,3)-2] ; 
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        for (auto index : lexical_inds){
//            steps.reset() ; 
//            arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) == end_ind + 1);
//            if (ref_step==1 || ref_step==2) { 
//                steps = temp_steps(arma::find(temp_steps == 0)); 
//            }
//            else if (ref_step==3) { 
//                steps = temp_steps(arma::find((temp_steps == 1) + (temp_steps==2))); 
//            }
//            //
//            if (steps.is_empty()) { 
//                continue; 
//            }
//            //
//            for (auto configuration : configurations) { 
//                if ( std::get<1>(configuration)==index+1) {
//                    for (int step : steps) {
//                        std::vector<int> comb = std::get<0>(configuration) ;
//                        comb.push_back( step );
//                        new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//    }
//
//    for (auto conf : configurations ) {
//        if ( std::get<1>(conf) == end_ind + 1) { 
//            std::vector comb( ket_vec.begin(), ket_vec.begin() + p - 1) ;  
//            comb.insert(comb.end(), std::get<0>(conf).begin(), std::get<0>(conf).end()) ;  
//            comb.insert(comb.end(), ket_vec.begin() + q , ket_vec.end()) ;
//            Configuration bra(comb) ; 
//            double factor = mec.one_body_coupling(bra, *this ,Epq) ; 
//            if (std::abs(factor) > 1e-8 ) { 
//                excitations.push_back( std::make_tuple(bra, factor) ); 
//            }
//        }    
//    }
//    // Sort greatest first
//    //std::sort(final.begin(), final.end(), std::less<Configuration>());
//    return excitations ; 
//} 
//
//
//std::vector<std::tuple<Configuration,double>> Configuration::apply_excitation(arma::imat &drt, const Eph &Epq) {
//    // Extract the hole and particle excitation indices
//    int p = (int) Epq.particle ; 
//    int q = (int) Epq.hole ;
//    
//    //assert(p<=q && "Function only locates D or R operator excitations");
//    std::vector<std::tuple<Configuration,double>> excitations ;  
//    MatrixElementCalculator mec;
//   
//    // Catch diagonal case 
//    // Diagonal case 
//    if ( p == q ) { 
//        double factor = mec.one_body_coupling(*this, *this ,Epq) ; 
//        if (std::abs(factor) > 1e-8 ) { 
//            excitations.push_back( std::make_tuple(*this, factor) ); 
//        }
//        return excitations ; 
//    }
//    
//    // L or R operator case 
//    // Find all the solutions that have the same outisde levels
//    std::vector<uint8_t> ket_vec = this->get_vec() ;
//    arma::irowvec start = {0,0,0} ; 
//    arma::irowvec end = {0,0,0} ; 
//    size_t start_ind = 0 ;
//    size_t end_ind = 0 ;
//    for (int i = 0 ; i < p-1 ; i ++) {
//        start += step_vecs.row(ket_vec[i]);
//    }
//    for (int i = 0 ; i < q ; i ++) {
//        end += step_vecs.row(ket_vec[i]);
//    }
//    bool sbool  = false; 
//    bool ebool = false; 
//    for (size_t i = 0 ; i < drt.n_rows ; i ++ ) {
//        if (!sbool && arma::all(drt.cols(0,2).row(i) == start)) {
//            start_ind = i ;
//            sbool = true;
//        }
//        if (!ebool && arma::all(drt.cols(0,2).row(i) == end)) {
//            end_ind = i ;
//            ebool = true;
//        }
//        if (sbool && ebool) { 
//            break; 
//        }
//    }
//    
//    // Store step vector and node lexical name  
//    std::vector< std::tuple< std::vector<int>, int> > configurations  = {std::make_tuple(std::vector<int> {}, start_ind+1)}; 
//    // Loop body
//    for ( int level = drt(start_ind,3) ; level < drt(end_ind,3) ; level++ ) { 
//        // Changing this line 
//        int ref_step = ket_vec[level-1] ;
//        arma::uvec allowed_steps = mec.one_body_drt(ref_step, level, Epq); 
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        // Extract current node positions 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        // Iterate over all current node positions 
//        for (auto index : lexical_inds){ 
//            std::vector<int> steps ;
//            if (level== head -1 ) { 
//                arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) == end_ind+1);
//            }
//            else { 
//                arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) != 0);
//            }
//            
//            // See if these are allowed steps 
//            for (auto s : temp_steps ) { 
//                if (arma::any(alllowed_steps == s)) { 
//                    steps.push_back(s); 
//                }
//            }
//            
//            //
//            if (steps.size() != 0 ) { 
//                for (auto configuration : configurations) { 
//                    if ( std::get<1>(configuration)==index+1) {
//                        for (auto step : steps ){ 
//                            std::vector<int> comb = std::get<0>(configuration) ;
//                            comb.push_back( step );
//                            new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                        }
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//    }
//
//    for (auto conf : configurations ) {
//        if ( std::get<1>(conf) == end_ind + 1) { 
//            std::vector comb( ket_vec.begin(), ket_vec.begin() + p - 1) ;  
//            comb.insert(comb.end(), std::get<0>(conf).begin(), std::get<0>(conf).end()) ;  
//            comb.insert(comb.end(), ket_vec.begin() + q , ket_vec.end()) ;
//            Configuration bra(comb) ; 
//            double factor = mec.one_body_coupling(bra, *this ,Epq) ; 
//            if (std::abs(factor) > 1e-8 ) { 
//                excitations.push_back( std::make_tuple(bra, factor) ); 
//            }
//        }    
//    }
//    // Sort greatest first
//    //std::sort(final.begin(), final.end(), std::less<Configuration>());
//    return excitations ; 
//}
//
//std::vector<std::tuple<Configuration,double>> Configuration::apply_excitation(arma::imat &drt, const Epphh &Epqrs) {
//    // Extract the hole and particle excitation indices
//    int i = (int) Epqrs.particle1 ; 
//    int j = (int) Epqrs.hole1 ;
//    int k = (int) Epqrs.particle2 ; 
//    int l = (int) Epqrs.hole2 ;
//    std::vector<std::tuple<Configuration,double>> excitations ;  
//    MatrixElementCalculator mec;
//  
//    // 
//    int head = std::max(i,j,k,l) ; 
//    int tail = std::min(i,j,k,l) ; 
//
//    // Catch diagonal case 
//    // Diagonal case
//    // what to return, we should probably refactor this alot... move loop body into a helper function and so on  
//    //if ( (i==j) || (k==l) ) { 
//    //    if ( (i==j) && (k==l)) { excitations = { std::make_tuple()}}
//    //    else if ( (i==j) ) { Eph Ekl(k,l) ; excitations = this->apply_excitations(drt,Ekl) ; }
//    //    else if ( (k==l) ) { Eph Eij(i,j) ; excitations = this->apply_excitations(drt,Eij) ; }
//    //}
//
//    // Find all the solutions that have the same outisde levels
//    // repeated block of code ... 
//    std::vector<uint8_t> ket_vec = this->get_vec() ;
//    arma::irowvec start = {0,0,0} ; 
//    arma::irowvec end = {0,0,0} ; 
//    size_t start_ind = 0 ;
//    size_t end_ind = 0 ;
//    for (int i = 0 ; i < tail-1 ; i ++) {
//        start += step_vecs.row(ket_vec[i]);
//    }
//    for (int i = 0 ; i < head ; i ++) {
//        end += step_vecs.row(ket_vec[i]);
//    }
//    bool sbool  = false; 
//    bool ebool = false; 
//    for (size_t i = 0 ; i < drt.n_rows ; i ++ ) {
//        if (!sbool && arma::all(drt.cols(0,2).row(i) == start)) {
//            start_ind = i ;
//            sbool = true;
//        }
//        if (!ebool && arma::all(drt.cols(0,2).row(i) == end)) {
//            end_ind = i ;
//            ebool = true;
//        }
//        if (sbool && ebool) { 
//            break; 
//        }
//    }
//   
//    // Need to extract S1 and S2 combined list 
//    std::set<int> S1;
//    std::set<int> S2;
//    for (int a = std::min({i,j,k,l}); a <= std::max({i,j,k,l}) ; a++){ 
//        // Overlapping range
//        if (( (a < std::max(k,l)) &&  (a >= std::min(k,l)) ) && ( (a < std::max(i,j)) &&  (a >= std::min(i,j)) ) ){ 
//            S1.insert(a);
//        }
//        else { 
//            S2.insert(a);
//        } 
//    }
//    // for (int level:
//
//
//    // Store step vector and node lexical name  
//    std::vector< std::tuple< std::vector<int>, int> > configurations  = {std::make_tuple(std::vector<int> {}, start_ind+1)}; 
//    // Loop body
//    for ( int level = drt(start_ind,3) ; level < drt(end_ind,3) ; level++ ) { 
//        // Changing this line 
//        int ref_step = ket_vec[level-1] ;
//        arma::uvec allowed_steps = mce.two_body_drt(ref_step, level, Epqrs); 
//        std::vector< std::tuple< std::vector<int>, int> > new_configurations ; 
//        std::vector<int> lexical_inds ; 
//        // Extract current node positions 
//        for ( auto conf : configurations ) { 
//            if (std::find(lexical_inds.begin(), lexical_inds.end(), std::get<1>(conf) - 1) == lexical_inds.end()) {
//                lexical_inds.push_back(std::get<1>(conf)-1); 
//            }
//        }
//
//        // Iterate over all current node positions 
//        for (auto index : lexical_inds){ 
//            std::vector<int> steps ;
//            if (level== head -1 ) { 
//                arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) == end_ind+1);
//            }
//            else { 
//                arma::uvec temp_steps = arma::find(drt.row(index).cols(4, drt.n_cols-1) != 0);
//            }
//            
//            // See if these are allowed steps 
//            for (auto s : temp_steps ) { 
//                if (arma::any(alllowed_steps == s)) { 
//                    steps.push_back(s); 
//                }
//            }
//            
//            //
//            if (steps.size() != 0 ) { 
//                for (auto configuration : configurations) { 
//                    if ( std::get<1>(configuration)==index+1) {
//                        for (auto step : steps ){ 
//                            std::vector<int> comb = std::get<0>(configuration) ;
//                            comb.push_back( step );
//                            new_configurations.push_back( std::make_tuple(comb, drt(index,4+step)));
//                        }
//                    }
//                }
//            }
//        }
//        configurations = new_configurations ; 
//    }
//
//    for (auto conf : configurations ) {
//        if ( std::get<1>(conf) == end_ind + 1) { 
//            std::vector comb( ket_vec.begin(), ket_vec.begin() + p - 1) ;  
//            comb.insert(comb.end(), std::get<0>(conf).begin(), std::get<0>(conf).end()) ;  
//            comb.insert(comb.end(), ket_vec.begin() + q , ket_vec.end()) ;
//            Configuration bra(comb) ; 
//            double factor = mec.one_body_coupling(bra, *this ,Epq) ; 
//            if (std::abs(factor) > 1e-8 ) { 
//                excitations.push_back( std::make_tuple(bra, factor) ); 
//            }
//        }    
//    }
//    // Sort greatest first
//    //std::sort(final.begin(), final.end(), std::less<Configuration>());
//    return excitations ; 
//}