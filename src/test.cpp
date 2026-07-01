#include "configuration.h"
#include <iostream>
#include <armadillo> 
int main () { 
    std::vector<uint8_t> drt1 = {3,1,2,0};
    Configuration conf1(drt1);
    std::vector<uint8_t> drt2 = {1,3,2,0};
    Configuration conf2(drt2);
    
    for(int i=0; i< conf1.m_nmo; i++) 
        std::cout << int(conf1.m_step_vec[i]) << " ";
    std::cout << std::endl;
    for(int i=0; i< conf2.m_nmo; i++) 
        std::cout << int(conf2.m_step_vec[i]) << " ";
    std::cout << std::endl;

    // Test the comparison operator
    // conf1 should be less than conf2 as more closed shell 
    std::cout << (conf1 < conf2) << std::endl;
    std::cout << (conf2 < conf1) << std::endl;

    arma::imat paldus1 = conf1.generate_paldus();
    arma::imat paldus2 = conf2.generate_paldus();
    paldus1.print("Paldus table for conf1:");
    paldus2.print("Paldus table for conf2:");


    return 0;
}
