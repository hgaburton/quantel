#include <omp.h>
#include <Eigen/Dense>
#include "shell_pair.h"

// set maximum precision for engine
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;
// use conservative screening method
constexpr auto screening_method = libint2::ScreeningMethod::SchwarzInf;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
using namespace libint2;

std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(
    const BasisSet& bs1, const BasisSet& _bs2, 
    const double threshold) 
{
    // Set basis sets equal if _bs2 is empty
    const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
    // Get number of shells in basis sets
    const auto nsh1 = bs1.size();
    const auto nsh2 = bs2.size();
    // Determine if basis sets are equivalent
    const auto bs1_equiv_bs2 = (&bs1 == &bs2);

    // Number of threads
    int nthreads = omp_get_max_threads();

    // construct the overlap integral engines
    using libint2::Engine;
    std::vector<Engine> engines;
    engines.reserve(nthreads);
    engines.emplace_back(Operator::overlap,
                         std::max(bs1.max_nprim(), bs2.max_nprim()),
                         std::max(bs1.max_l(), bs2.max_l()), 0);
    engines[0].set_precision(0.);
    for (size_t i = 1; i != nthreads; ++i) 
    {
        engines.push_back(engines[0]);
    }

    std::cout << "Computing non-negligible shell pairs ... ";

    libint2::Timers<1> timer;
    timer.set_now_overhead(25);
    timer.start(0);

    // Initialise shell pair list
    shellpair_list_t splist;
    // Initialise mutex for thread safety
    std::mutex mx;

    // Compute 
    #pragma omp parallel
    {
        // Setup thread-safe engine
        auto thread_id = omp_get_thread_num();
        auto &engine = engines[thread_id];
        const auto &buf = engine.results();

        // loop over permutationally-unique set of shells
        for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) 
        {
            // Thread-safe add shell-pair if not in list.
            mx.lock();
            if (splist.find(s1) == splist.end())
                splist.insert(std::make_pair(s1, std::vector<size_t>()));
            mx.unlock();

            // Number of basis functions in this shell
            auto n1 = bs1[s1].size();
            auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
            // Loop over other basis functions
            for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) 
            {
                // Skip if not in this thread batch
                if (s12 % nthreads != thread_id) continue;
                auto on_same_center = (bs1[s1].O == bs2[s2].O);
                bool significant = on_same_center;
                // Test overlap to establish if shell-pair is significant
                if (not on_same_center) 
                {
                    auto n2 = bs2[s2].size();
                    engines[thread_id].compute(bs1[s1], bs2[s2]);
                    Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
                    auto norm = buf_mat.norm();
                    significant = (norm >= threshold);
                }
                // Add shell-pair if significant
                if (significant) 
                {
                    mx.lock();
                    splist[s1].emplace_back(s2);
                    mx.unlock();
                }
            }
        }
    }

    // Sort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
    #pragma omp parallel
    {
        auto thread_id = omp_get_thread_num();
        for (auto s1 = 0l; s1 != nsh1; ++s1) 
        {
            if (s1 % nthreads == thread_id) 
            {
                auto& list = splist[s1];
                std::sort(list.begin(), list.end());
            }
        }
    }

    // Compute shellpair data assuming data used for Coulomb ints
    const auto ln_max_engine_precision = std::log(max_engine_precision);
    for (auto&& eng : engines) eng.set(Operator::coulomb);
    shellpair_data_t spdata(splist.size());

    // Setup the shellpair data with appropriate screening method    
    #pragma omp parallel
    {
        auto thread_id = omp_get_thread_num();
        // Define lambda function to compute Schwarz screeing of shell pair primitives
        auto schwarz_factor_evaluator = [&](const Shell& s1, size_t p1,
                                            const Shell& s2, size_t p2) -> double 
        {
            auto& engine = engines[thread_id];
            auto& buf = engine.results();
            auto ps1 = s1.extract_primitive(p1, false);
            auto ps2 = s2.extract_primitive(p2, false);
            const auto n12 = ps1.size() * ps2.size();
            engine.compute(ps1, ps2, ps1, ps2);
            if (buf[0]) 
            {
                Eigen::Map<const Matrix> buf_mat(buf[0], n12, n12);
                auto norm2 = (screening_method == ScreeningMethod::SchwarzInf)
                                ? buf_mat.lpNorm<Eigen::Infinity>()
                                : buf_mat.norm();
                return std::sqrt(norm2);
            } 
            else return 0.0;
        };
 
        // Loop over basis sets
        for (auto s1 = 0l; s1 != nsh1; ++s1) 
        {
            if (s1 % nthreads == thread_id) 
            {
                for (const auto& s2 : splist[s1]) 
                {
                    if (screening_method == ScreeningMethod::Original ||
                        screening_method == ScreeningMethod::Conservative)
                    {
                        spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(
                            bs1[s1], bs2[s2], ln_max_engine_precision, screening_method));
                    }
                    else 
                    {  // Schwarz screening of primitives
                        spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(
                            bs1[s1], bs2[s2], ln_max_engine_precision, screening_method,
                            schwarz_factor_evaluator));
                    }
                }
            }
        }
    }

    timer.stop(0);
    std::cout << "done (" << timer.read(0) << " s)" << std::endl;

    return std::make_tuple(splist, spdata);
}