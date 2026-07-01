#include <iostream> 
#include <vector> 
#include <set> 
#include "../../external/armadillo/include/armadillo" 
#include <algorithm>

bool contains(const std::vector<std::string>& vec, const std::string& val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

inline const arma::imat step_vecs = { 
{0,0,1},
{0,1,0},
{1,-1,1},
{1,0,0}
};



template <typename StepFunc>
void shared_body(int a, int b, StepFunc step_func) {
    int c = a - b ; 
    int d = a + b ; 
    std::cout << "int c " << std::to_string(c) << std::endl ; 
    std::cout << "int d " << std::to_string(d) << std::endl ; 
    int e = step_func(c,d) ; 
    std::cout << "Result (e) " << std::to_string(e) << std::endl ; 
    return;   
}  

int func1( int a , int b) { 
    int c = a + b ; 
    return c ; 
} 

int func2( int a , int b) { 
    int c = a*b ; 
    return c ; 
}

int main() { 
    
    //std::vector<int> S1 = { 1,2,3,4} ; 
    //std::vector<int> S2 = { 7,8,9,10} ; 
    //std::cout << "Printing list... " << std::endl ;
    //std::vector<int> combined ; 
    //for (auto a :std::vector<int> S1+S2) { std::cout << " " << std::to_string(a) << " "; } 
 


    shared_body(1,2, [&](int a, int b) { return func1(a, b); } ) ;  
    shared_body(1,2, [&](int a, int b) { return func2(a, b); } ) ;  
    return 1 ; 
} 
