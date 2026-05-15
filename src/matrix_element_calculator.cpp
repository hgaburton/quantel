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
        if ( arma::all( bra_paldus.row(a) != ket_paldus.row(a) ) ) { 
            return 0.0 ; 
        }
    }

    for (int a = head + 1 ; a <= nmo ; a ++ ) { 
        if ( arma::all( bra_paldus.row(a) != ket_paldus.row(a) ) ) { 
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



double MatrixElementCalculator::two_body_coupling( const Configuration &bra, const Configuration &ket, const Epphh &Epphh ) const { 
    // Check same number of electrons, orbitals and S 
    if ( (bra.m_nmo != ket.m_nmo) || (bra.m_nelec != ket.m_nelec) || (bra.m_totspin != ket.m_totspin)  ) { 
        std::cout << "Different N, n or S values" << std::endl ; 
        return 0.0 ; 
    }
    const size_t nmo = bra.m_nmo ;
    const arma::imat bra_paldus = bra.generate_paldus() ; 
    const arma::imat ket_paldus = ket.generate_paldus() ;

    // Make sure both sets of indices are in range 
    int i = (int) Epphh.particle1 ; 
    int j = (int) Epphh.hole1 ; 
    int k = (int) Epphh.particle2 ; 
    int l = (int) Epphh.hole2; 
    assert( std::max({i,j,k,l}) <= bra.m_nmo ) ;
    assert( std::min({i,j,k,l}) > 0 ) ; 
    // Check path outside loop 
    for (int a = 0 ; a < nmo ; a++ ) {
        // iterate over the full loop  
        if ( ( a > std::max({i,j,k}) ) || ( a < std::min({i,j,k}) )  ) {
            // if a is outside the "true_loop" 
            if ( arma::all( bra_paldus.row(a) != ket_paldus.row(a) ) ) { 
                return 0.0 ; 
            }
        } 
    }
    
    // Deal with possible number operators 
    double matrix_element = 0.0 ;
    if ( i==j || k==l ){  
        Eph E1={ (size_t) i,(size_t) j}; 
        Eph E2={(size_t)k, (size_t)l}; 
        if (i==j && k==l) {         
        matrix_element += one_body_coupling(bra, ket, E1 )*one_body_coupling(bra, ket, E2);
        }
        else if (i==j) {  
        matrix_element += one_body_coupling(bra, bra, E1)*one_body_coupling(bra, ket, E2);
        }
        else if (k==l) {  
        matrix_element += one_body_coupling(bra, ket, E1)*one_body_coupling(ket, ket, E2);
        }
        if (j == k) {
            Eph E3={(size_t)i,(size_t)l};              
            matrix_element -= one_body_coupling(bra, ket, E3);
        }
        return matrix_element ; 
    }
    
    // Choose if R, L - there shouldnt be any D values left
    std::vector<std::string> ab_classes(2, "") ;  
    std::vector<int> RorLs(2); 
    if (i - j > 0) {ab_classes[0] = "L"; RorLs[0] = 1 ;}
    else if (i - j < 0) {ab_classes[0] = "R"; RorLs[0] = 0 ;}
    if (k - l > 0) {ab_classes[1] = "L"; RorLs[1] = 1 ;}
    else if (k - l < 0) {ab_classes[1] = "R"; RorLs[1] = 0 ;}
 
    std::set<int> S1;
    std::set<int> S2a;
    std::set<int> S2b; 
    for (int a = std::min({i,j,k,l}); a < std::max({i,j,k,l}) ; a++){ 
        // Overlapping range 
        if (( (a <= std::max(k,l)) &&  (a >= std::min(k,l)) ) && ( (a <= std::max(i,j)) &&  (a >= std::min(i,j)) ) ){ 
            S1.insert(a);
        } 
        // Non-overlapping range 
        if ( (a > std::max(k,l)) &&  (a < std::min(k,l)) ) { 
            S2a.insert(a);
        }
        else if  ( (a > std::max(i,j)) &&  (a < std::min(i,j)) ) { 
            S2b.insert(a);
        }
    }
    // Construct overlapping and non-overlapping ranges 
    //std::set_union(seta.begin(), seta.end(), setb.begin(), setb.end(), std::inserter(Sunion), Sunion.begin()); 
    //std::set_intersection(seta.begin(), seta.end(), setb.begin(), setb.end(), std::inserter(S1), S1.begin());
    //std::set_difference(Sunion.begin(), Sunion.end(), S1.begin(), S1.end(), std::inserter(S2), S2.begin() ) ;  

     
    // non overlapping range 
    // There is a better way to do this right - we should be able to do the one body fragment in this way
    matrix_element = 1.0 ; 
    // could we just add these
    std::vector<std::set<int>> S2 = { S2a, S2b} ;
    std::vector<int> tail_inds = { std::min(i,j), std::min(k,l)} ;
    std::vector<int> head_inds = { std::max(i,j), std::max(k,l)} ;
    std::cout << "tail inds " << " " ; 
    for (const int a : tail_inds) { 
        std::cout << std::to_string(a) << " " ; 
    }
    std::cout <<  std::endl ; 
    std::cout << "head inds " << " " ; 
    for (const int a : head_inds) { 
        std::cout << std::to_string(a) << " " ; 
    }
    std::cout <<  std::endl ; 
    
    std::cout << "S2a " << " " ; 
    for (const int a : S2a) { 
        std::cout << std::to_string(a) << " " ; 
    }
    std::cout << std::endl ; 
    std::cout << "S2b " << " " ; 
    for (const int a : S2b) { 
        std::cout << std::to_string(a) << " " ; 
    }
    std::cout << std::endl ; 
    
    for (int a = 0 ; a < 2 ; a++ ){
        int tail_ind = tail_inds[a]; 
        int head_ind = tail_inds[a];    
        int RorL = RorLs[a] ;

        for (const int ind : S2[a]) { 
            const int d1 = bra.m_step_vec[ind-1]; 
            const int d2 = ket.m_step_vec[ind-1];
            const int b = ket_paldus(ind, 1) ; 
            const int delta_b = ket_paldus( ind, 1) - bra_paldus(ind, 1); 
            matrix_element *= one_body_fragment( ind, d1, d2, b, delta_b, tail_ind, head_ind, RorL) ;
            if (matrix_element == 0.0) { 
                return matrix_element ; 
            }
        }
    }    


    std::cout << "S1 " << " " ; 
    for (const int a : S1) { 
        std::cout << std::to_string(a) << " " ; 
    }
    std::cout << std::endl ; 
    

    // overlapping range 
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
        else if (contains(operators, "tR") && contains(operators, "hR")) {
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
            if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
            x0 *= tb_table_four[d1][d2][0][get_Dind(delta_b)][0](b);
            x1 *= tb_table_four[d1][d2][0][get_Dind(delta_b)][1](b);
        }
        else if (operators[0] == "R" && operators[1] == "tR") {
            if (std::abs(delta_b) != 2 && std::abs(delta_b) != 0) { return 0.0; }
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
            std::cerr << "Fell through Error: " << operators[0] << " " << operators[1] << std::endl;
        }        
    }
    matrix_element  *= x0 + x1 ; 
    return matrix_element ; 
}