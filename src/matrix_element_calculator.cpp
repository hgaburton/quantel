#include "configuration.h"
#include "matrix_element_calculator.h"
#include "lookup_table.h"
#include "excitation.h"
using namespace LookupTables ; 
#include <armadillo> 
#include <vector> 
#include <iostream> 
#include <cassert>
#include <set>

// define helper functions 
bool MatrixElementCalculator::contains(const std::vector<std::string>& vec, const std::string& val) const {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

std::string MatrixElementCalculator::HeadsOrTails(const int &ind, const int &i, const int &j ) const { 
    if (ind == std::max(i,j)) { return "h";}
    else if ( ind == std::min(i,j)) {return "t"; }
    else { return "" ;}
}

int MatrixElementCalculator::get_Dind(const int &delta_b) const {
    if (delta_b == -1 || delta_b == -2) {return 0 ;}
    else if (delta_b == 1 || delta_b == 0 ) {return 1 ;}
    else if (delta_b == 2) {return 2 ;}
    else{std::cerr << "An error occurred " << std:: endl ; return 500 ;}
}

double MatrixElementCalculator::one_body_coupling(const Configuration &bra, const Configuration &ket, const Eph &Eph) const { 
    // Calculate the one body coupling matrix element 
    // Check same number of electrons, orbitals and S 
    if ( (bra.m_nmo != ket.m_nmo) || (bra.m_nelec != ket.m_nelec) || (bra.m_totspin != ket.m_totspin)  ) { 
        std::cout << "Different N, n or S values" << std::endl ; 
        return 0.0 ; 
    }

    const size_t nmo = bra.m_nmo ;
    const arma::imat bra_paldus = bra.generate_paldus() ; 
    const arma::imat ket_paldus = ket.generate_paldus() ;

    // Define excitation index loop
    int i = (int) Eph.particle ; 
    int j = (int) Eph.hole ; 
    int head = std::max(i,j) ; 
    int tail = std::min(i,j) ; 
    // Make sure both indices are in range 
    // KEY POINT!!! 
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    // treating these as 1 indexed!  
    assert( head <= bra.m_nmo ) ; 
    assert( tail > 0 ) ;
    
    // Check outside loops are the same  
    for (int a = 0 ; a < tail ; a ++ ) { 
        if (!arma::all( bra_paldus.row(a) == ket_paldus.row(a) ) ) {
            return 0.0 ; 
        }
    }
    //for (int a = head + 1 ; a <= nmo ; a ++ ) { 
    for (int a = head ; a <= nmo ; a ++ ) { 
        if (!arma::all( bra_paldus.row(a) == ket_paldus.row(a) ) ) {
            return 0.0 ; 
        }
    }

    // Calc loop values 
    // Diagonals 
    if ( Eph.hole == Eph.particle ) { 
        // i - 1, as i = 1 is at bra[0] position
        // paldus table has top row 0,0,0 so index with i 
        return ob_table_one[bra.m_step_vec[i-1] ][ ket.m_step_vec[i-1] ][0]( ket_paldus((int) Eph.particle,1) );
    }

    // Off-diagonals 
    int RorL =  ( i - j > 0 );
    double matrix_element = 1.0 ; 
    for ( int a = tail ; a <= head ; a++) {
        const int d1 = bra.m_step_vec[a-1]; 
        const int d2 = ket.m_step_vec[a-1];
        const int b = ket_paldus(a, 1) ; 
        const int delta_b = ket_paldus( a, 1) - bra_paldus(a, 1);  
        matrix_element *= one_body_fragment(a, d1, d2, b, delta_b, head, tail, RorL); 
        if (matrix_element == 0.0) { 
            return 0.0 ; 
        }
    } 
    return matrix_element; 
}

//double MatrixElementCalculator::one_body_fragment(const Configuration &bra, const Configuration &ket, const int &level) const {  
double MatrixElementCalculator::one_body_fragment(const int &level, const int &d1, const int &d2, const int &b, const int &delta_b, const int &head, const int &tail, const int &RorL ) const { 
    double factor ; 
    if ( level == head) { 
    // Loop head 
        factor = ob_table_one[d1][d2][RorL + 1 ](b);
    }
    else if (std::abs(delta_b) != 1) { 
        factor = 0.0 ;  
    }
    else if (level == tail) { 
        // Loop tail 
        factor = ob_table_one[d1][d2][RorL+3](b);
    }
    else { 
        int Dind = (delta_b == -1) ? 0 : 1 ;
        factor = ob_table_two[d1][d2][RorL][Dind](b);
    }
    return factor;
}


double MatrixElementCalculator::two_body_coupling( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs ) const { 
    // Check same number of electrons, orbitals and S 
    if ( (bra.m_nmo != ket.m_nmo) || (bra.m_nelec != ket.m_nelec) || (bra.m_totspin != ket.m_totspin)  ) { 
        std::cout << "Different N, n or S values" << std::endl ; 
        return 0.0 ; 
    }
    const size_t nmo = bra.m_nmo ;
    const arma::imat bra_paldus = bra.generate_paldus() ; 
    const arma::imat ket_paldus = ket.generate_paldus() ;

    // Make sure both sets of indices are in range 
    int i = (int) Epqrs.particle1 ; 
    int j = (int) Epqrs.hole1 ; 
    int k = (int) Epqrs.particle2 ; 
    int l = (int) Epqrs.hole2 ;
    
    // Print
    //std::cout << "-------------" << std::endl ;  
    //std::cout << "e ( " << std::to_string(i) << ", " << std::to_string(k) << ", " << ", " << std::to_string(j) << ", " << std::to_string(l) << " )" << std::endl ; 

    assert( std::max({i,j,k,l}) <= bra.m_nmo ) ;
    assert( std::min({i,j,k,l}) > 0 ) ; 
    
    // Check path outside loop 
    std::vector<int> tail_inds = {std::min(i,j),std::min(k,l)} ;
    std::vector<int> head_inds = {std::max(i,j),std::max(k,l)} ;
    for (int a = 1 ; a <= nmo ; a ++ ) { 
        if (!((( a < head_inds[0]) && (a >= tail_inds[0])) || (( a < head_inds[1]) && (a >= tail_inds[1])))) { 
            if (!arma::all( bra_paldus.row(a) == ket_paldus.row(a) ) ) {
                //std::cout << "No overlap outside loop! " << std::endl ;  
                return 0.0 ; 
            }
        }
    }
    
    // Deal with possible number operators 
    double matrix_element = 0.0 ;
    if ( i==j || k==l ){  
        if (i==j && k==l) {         
            matrix_element += one_body_coupling(bra, ket, {(size_t) i , (size_t) i} )*one_body_coupling(bra, ket, {(size_t) k , (size_t) k});
            //std::cout << "Diag 1" << std::endl ; 
        }
        else if (i==j) {  
            matrix_element += one_body_coupling(bra, bra, {(size_t) i , (size_t) i} )*one_body_coupling(bra, ket, {(size_t) k , (size_t) l});
            //std::cout << "Diag 2" << std::endl ; 
        }
        else if (k==l) {  
            matrix_element += one_body_coupling(bra, ket, {(size_t) i , (size_t) j} )*one_body_coupling(ket, ket, {(size_t) k , (size_t) k});
            //std::cout << "Diag 3" << std::endl ; 
        }
        if (j == k) {
            matrix_element -= one_body_coupling(bra, ket, {(size_t) i , (size_t) l} ) ;
            //std::cout << "Diag 4" << std::endl ; 
        }
        return matrix_element ; 
    }
    

    // S1 and S2 loops
    std::vector<int> S1 ;
    std::vector<int> S2 ;
    // Nope this is not true since we need to make sure that these are in the correct loops
    for (int a = std::min({i,j,k,l}); a <= std::max({i,j,k,l}) ; a++){ 
        // Overlapping range
        if (( (a <= head_inds[1]) &&  (a >= tail_inds[1]) ) && ( (a <= head_inds[0]) &&  (a >= tail_inds[0]) ) ){ 
            S1.push_back(a);
        }
        else if (( (a <= head_inds[1]) &&  (a >= tail_inds[1]) ) || ( (a <= head_inds[0]) &&  (a >= tail_inds[0]) ) ){ 
            S2.push_back(a);
        } 
    }
    
    // Choose if R, L - there shouldnt be any D values left
    std::vector<std::string> ab_classes(2, "") ;  
    std::vector<int> RorLs(2); 
    if (i - j > 0) {ab_classes[0] = "L"; RorLs[0] = 1 ;}
    else if (i - j < 0) {ab_classes[0] = "R"; RorLs[0] = 0 ;}
    if (k - l > 0) {ab_classes[1] = "L"; RorLs[1] = 1 ;}
    else if (k - l < 0) {ab_classes[1] = "R"; RorLs[1] = 0 ;}
    

    // toggle print statements 
    //std::cout << "tail inds " << " " ; 
    //for (const int a : tail_inds) { 
    //    std::cout << std::to_string(a) << " " ; 
    //}
    //std::cout <<  std::endl ; 
    //std::cout << "head inds " << " " ; 
    //for (const int a : head_inds) { 
    //    std::cout << std::to_string(a) << " " ; 
    //}
    //std::cout <<  std::endl ; 
    //non overlapping range 
    //There is a better way to do this right - we should be able to do the one body fragment in this way
    //std::cout << "S2 loop " << std::endl ; 
    matrix_element = 1.0 ; 
    for (int ind : S2  ){
        //std::cout << "S2 val: " << std::to_string(ind) << std::endl;
        int tail_ind ; 
        int head_ind ; 
        int RorL ; 
        
        // identitfing which loop it belongs to 
        if ( (ind >= std::min(i,j)) && (ind <= std::max(i,j))) { 
            tail_ind = tail_inds[0]; 
            head_ind = head_inds[0];    
            RorL = RorLs[0] ;
        }
        else if ( (ind >= std::min(k,l)) && (ind <= std::max(k,l))) { 
            tail_ind = tail_inds[1]; 
            head_ind = head_inds[1];    
            RorL = RorLs[1] ;
        } 
        //std::cout << "inl: tail,head,RorL " << std::to_string(tail_ind) << " " << std::to_string(head_ind) << " " <<  std::to_string(RorL) << std::endl ; 

         const int d1 = bra.m_step_vec[ind-1]; 
         const int d2 = ket.m_step_vec[ind-1];
         const int b = ket_paldus(ind, 1) ; 
         const int delta_b = ket_paldus( ind, 1) - bra_paldus(ind, 1); 
         //std::cout << " d1, d2, b, deltab " << std::to_string(d1) << " " << std::to_string(d2) << " " << std::to_string(b)<< " " << std::to_string(delta_b) << std::endl; 
         matrix_element *= one_body_fragment( ind, d1, d2, b, delta_b, head_ind, tail_ind, RorL) ;
         //std::cout << "Matrix element: " << std::to_string(matrix_element) << std::endl ; 
         if (matrix_element == 0.0) { 
             return matrix_element ; 
        }
    } 
   
    if (S1.size()==0) {
        return matrix_element ; 
    }

    // toggle print statements 
    //std::cout << "S1 " ;
    //for (auto a : S1 ) { 
    //    std::cout << std::to_string(a) << " " ; 
    //}  
    //std::cout << " end " << std::endl; 
    // overlapping range 

    if (S1.size()==1) { 
        std::vector<std::string> operators(2); 
        operators[0] = HeadsOrTails(S1[0], i, j) + ab_classes[0]; 
        operators[1] = HeadsOrTails(S1[0], k, l) + ab_classes[1];
        const int d1 = bra.m_step_vec[S1[0]-1]; 
        const int d2 = ket.m_step_vec[S1[0]-1];
        const int b = ket_paldus(S1[0], 1) ; 
        const int delta_b = ket_paldus( S1[0], 1) - bra_paldus( S1[0], 1); 

        // Check if its any of the loop ending operators 
        if (contains(operators, "tR") && contains(operators, "hR")) {
            if (std::abs(delta_b) != 1) { return 0.0; }
            matrix_element *= tb_table_two[d1][d2][0][get_Dind(delta_b)](b);
        }
        else if (contains(operators, "tL") && contains(operators, "hL")) {
            if (std::abs(delta_b) != 1) { return 0.0; }
            matrix_element *= tb_table_two[d1][d2][1][get_Dind(delta_b)](b);
        }
        else if (contains(operators, "hR") && contains(operators, "tL")) {
            if (std::abs(delta_b) != 1) { return 0.0; }
            matrix_element *= tb_table_two[d1][d2][2][get_Dind(delta_b)](b);
        }
        else if (contains(operators, "tR") && contains(operators, "hL")) {
            if (std::abs(delta_b) != 1) { return 0.0; }
            matrix_element *= tb_table_two[d1][d2][3][get_Dind(delta_b)](b);
        }
        else {
            std::cerr << "Fell through Error1: " << operators[0] << " " << operators[1] << std::endl;
        }
        //std::cout << "S1 index " << std::to_string(S1[0]) << " MatEl: " << std::to_string(matrix_element) << std::endl ;         
        //std::cout << " d1, d2, b, deltab " << std::to_string(d1) << " " << std::to_string(d2) << " " << std::to_string(b)<< " " << std::to_string(delta_b) << std::endl; 
        //std::cout << " Operators: " << operators[0] << " " << operators[1] << std::endl ; 
        return matrix_element; 
    }
    else {
        // x0 and x1 contributions 
        double x0 = 1.0 ; 
        double x1 = 1.0 ; 
        for ( const int ind : S1 ) { 
            std::vector<std::string> operators(2); 
            operators[0] = HeadsOrTails(ind, i, j) + ab_classes[0]; 
            operators[1] = HeadsOrTails(ind, k, l) + ab_classes[1];
            const int d1 = bra.m_step_vec[ind-1]; 
            const int d2 = ket.m_step_vec[ind-1];
            const int b = ket_paldus(ind, 1) ; 
            const int delta_b = ket_paldus( ind, 1) - bra_paldus( ind, 1); 

            if ((operators[0] == "hR" && operators[1] == "hR") || (operators[0] == "tL" && operators[1] == "tL")) {
                x0 *= tb_table_one[d1][d2][0][0](b);
                x1 *= tb_table_one[d1][d2][0][1](b);
            }
            else if ((operators[0] == "tR" && operators[1] == "tR") || (operators[0] == "hL" && operators[1] == "hL")) {
                x0 *= tb_table_one[d1][d2][1][0](b);
                x1 *= tb_table_one[d1][d2][1][1](b);
            }
            else if (contains(operators, "hR") && contains(operators, "hL")) {
                x0 *= tb_table_one[d1][d2][2][0](b);
                x1 *= tb_table_one[d1][d2][2][1](b);
            }
            else if (contains(operators, "tR") && contains(operators, "tL")) {
                x0 *= tb_table_one[d1][d2][3][0](b);
                x1 *= tb_table_one[d1][d2][3][1](b);
            }
            else if (operators[0] == "R" && operators[1] == "hR") {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][0][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][0][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "hR" && operators[1] == "R") {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][1][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][1][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "hL" && operators[1] == "L") {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][2][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][2][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "L" && operators[1] == "hL") {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][3][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][3][get_Dind(delta_b)][1](b);
            }
            else if (contains(operators, "hR") && contains(operators, "L")) {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][4][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][4][get_Dind(delta_b)][1](b);
            }
            else if (contains(operators, "R") && contains(operators, "hL")) {
                if (std::abs(delta_b) != 1) { return 0.0; }
                x0 *= tb_table_three[d1][d2][5][get_Dind(delta_b)][0](b);
                x1 *= tb_table_three[d1][d2][5][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "tR" && operators[1] == "R") {
                if (std::abs(delta_b) != 2.0 && std::abs(delta_b) != 0.0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][0][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][0][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "R" && operators[1] == "tR") {
                if (std::abs(delta_b) != 2.0 && std::abs(delta_b) != 0.0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][1][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][1][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "L" && operators[1] == "tL") {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][2][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][2][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "tL" && operators[1] == "L") {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][3][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][3][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "R" && operators[1] == "R") {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][4][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][4][get_Dind(delta_b)][1](b);
            }
            else if (operators[0] == "L" && operators[1] == "L") {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][5][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][5][get_Dind(delta_b)][1](b);
            }
            else if (contains(operators, "R") && contains(operators, "tL")) {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][6][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][6][get_Dind(delta_b)][1](b);
            }
            else if (contains(operators, "tR") && contains(operators, "L")) {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][7][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][7][get_Dind(delta_b)][1](b);
            }
            else if (contains(operators, "R") && contains(operators, "L")) {
                if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
                x0 *= tb_table_four[d1][d2][8][get_Dind(delta_b)][0](b);
                x1 *= tb_table_four[d1][d2][8][get_Dind(delta_b)][1](b);
            }
            else {
                std::cerr << "Fell through Error2: " << operators[0] << " " << operators[1] << std::endl;
            }
            //std::cout << "S1 index " << std::to_string(ind) << " MatEl x0: " << std::to_string(x0) << " x1: " << std::to_string(x1) << std::endl ;         
            //std::cout << " d1, d2, b, deltab " << std::to_string(d1) << " " << std::to_string(d2) << " " << std::to_string(b)<< " " << std::to_string(delta_b) << std::endl; 
            //std::cout << " Operators: " << operators[0] << " " << operators[1] << std::endl ; 
    
        }
        matrix_element  *= x0 + x1 ; 
        return matrix_element ;
    } 
}


double MatrixElementCalculator::resolve_two_body_matrix_element( const Configuration &bra, const Configuration &ket, const Epphh &Epqrs, const std::vector<Configuration> &basis) const { 
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
    MatrixElementCalculator mb ;  
    for (auto config  : basis) {
        matrix_element += mb.one_body_coupling(bra, config, {i,j})*mb.one_body_coupling(config, ket, {k,l}) ;   
    }
    if (j==k) { 
        matrix_element -= mb.one_body_coupling(bra, ket,{i,l}) ;  
    }
    return matrix_element ; 
}
