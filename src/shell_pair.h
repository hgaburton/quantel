#ifndef SHELL_PAIR_H
#define SHELL_PAIR_H

#include <unordered_map>
#include <libint2.hpp>

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;

/// computes non-negligible shell pair list; shells \c i and \c j form a
/// non-negligible pair if they share a center or the Frobenius norm of their overlap is
/// greater than threshold
std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(
    const libint2::BasisSet& bs1, 
    const libint2::BasisSet& _bs2 = libint2::BasisSet(), 
    const double threshold=1e-12);

#endif // SHELL_PAIR_H