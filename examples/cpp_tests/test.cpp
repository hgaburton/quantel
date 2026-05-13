#include <iostream> 
#include <vector> 
#include <set> 
#include "../../external/armadillo/include/armadillo" 
#include <algorithm>

bool contains(const std::vector<std::string>& vec, const std::string& val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}
int main() { 
    //arma::imat A = { {5,4},{4,3} } ; 
    //arma::imat B = { {5,4},{1,2} } ;
    //A.print("A matrix");  
    //B.print("B matrix"); 

    //for (int a = 0 ; a < 2 ; a ++ ) { 
    //    for (int b = 0 ; b < 2 ; b ++ ) {
    //        std::cout << "( "<< a << " , " << b <<" ) " << A(a,b) << std::endl ; 
    //    } 
    //}

    //A.row(0).print("A mat ; Row 0"); 
    //B.row(1).print("B mat ; Row 0"); 

    //if ( arma::all(A.row(0) !=  B.row(0)) ) { 
    //    std::cout << "diff first row" << std::endl ; 
    //}
    //else { 
    //    std::cout << "same first row" << std::endl ; 
    //}
    std::vector<std::string> left = { "hR", "hR" } ;
    std::cout << "output: " << std::to_string(contains(left, "hR")) << std::endl ; 
    if (contains( left, "hR" ))  { 
        std::cout << "Contains = True " << std::endl ; 
    } 
    return 1 ; 
} 
