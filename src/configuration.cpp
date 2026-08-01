#include "configuration.h"
#include "excitation.h" 
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

