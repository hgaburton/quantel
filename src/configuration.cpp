#include "configuration.h" 
#include <armadillo> 
#include <vector>
#include <iostream>

arma::imat step_vecs = { 
    {0, 0, 1}, 
    {0, 1, 0}, 
    {1,-1, 1}, 
    {1, 0, 0} 
};

arma::imat Configuration::generate_paldus() const {
    // Implementation for generating Paldus table representation
    arma::imat paldus_table(m_nmo+1, 3, arma::fill::zeros);
    for (int i = 0; i < m_nmo ; i++){ 
        paldus_table.row(i+1) = paldus_table.row(i) + step_vecs.row(m_step_vec[i]); 
    }
    return paldus_table;
}

arma::imat Configuration::construct_drt() const { 
    // Construct table of all possible rows in Paldus table to construct the final row 
    // top row of the table 
    int a = ( m_nelec - 2*m_totspin)/2 ; 
    int b = 2 * m_totspin ; 
    int c = m_nmo - a - b; 
    
    std::vector<std::vector<int>> soln_array ; 
    soln_array.push_back( { a, b, c, m_nmo, 0, 0, 0, 0} ) ; 
    std::vector<size_t> connected_solns = { 0 };  
    for (int i = m_nmo ; i >= 0 ; i--) { 

        size_t start = soln_array.size() ;
        std::vector<std::vector<int>> temp_solns; 
        std::vector<size_t> temp_inds ; 
        
        for (size_t j = 0 ; j < connected_solns.size() ; j++) { 
            size_t prev = connected_solns[j] ; 
            for (uint8_t d = 0 ; d < 4 ; d++) { 
                // Connect by one of the step vectors
                std::vector<int> current(8);
                current[0] = soln_array[prev][0] - step_vecs(d, 0);
                current[1] = soln_array[prev][1] - step_vecs(d, 1);
                current[2] = soln_array[prev][2] - step_vecs(d, 2);
                current[3] = i;
                // so we save solution name rather than index
                // this is such that 0 == "None" stand-in 
                current[4+d] = prev + 1 ; 
                if ( (current[0]>=0) && (current[1]>=0) && (current[2]>=0) ) { 
                    bool inlist = false; 
                    
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
        connected_solns = temp_inds ; 
    }

    arma::imat distinct_row_table(soln_array.size(), 8);
    for (size_t i = 0; i < soln_array.size(); i++) {
        for (size_t j = 0; j < 8; j++) {
            distinct_row_table(i, j) = soln_array[i][j];
        }
    }
    return distinct_row_table; 
}

