#include "configuration.h"
#include <iostream>

int main () { 

    std::vector<uint8_t> drt1 = {3,2,1,0};
    std::vector<uint8_t> drt2 = {3,1,2,0};
    Configuration conf1(drt1);
    Configuration conf2(drt2);
    
    for(int i=0; i< conf1.m_nmo; i++) 
        std::cout << int(conf1.m_drt[i]) << " ";
    std::cout << std::endl;
    for(int i=0; i< conf2.m_nmo; i++) 
        std::cout << int(conf2.m_drt[i]) << " ";
    std::cout << std::endl;

    // Test the comparison operator
    // conf1 should be less than conf2 as more closed shell 
    std::cout << (conf1 < conf2) << std::endl;
    std::cout << (conf2 < conf1) << std::endl;

    armadillo::mat paldus1 = conf1.generate_paldus(conf1.m_drt);
    armadillo::mat paldus2 = conf2.generate_paldus(conf2.m_drt);
    paldus1.print("Paldus table for conf1:");
    paldus2.print("Paldus table for conf2:");


    return 0;
}