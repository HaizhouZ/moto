#define CATCH_CONFIG_MAIN
#include <Eigen/Cholesky>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/utils/blasfeo_factorization.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

TEST_CASE("llt_test") {
    using namespace moto;
    size_t n_trials = 100;
    std::vector<matrix> As(n_trials), bs(n_trials), x_refs(n_trials), xs(n_trials);
    std::vector<size_t> dims(n_trials, 12), dim2s(n_trials, 37);

    // Generate random matrices
    for (size_t i = 0; i < n_trials; i++) {
        As[i] = matrix::Random(dims[i], dims[i]);
        As[i] = As[i].transpose() * As[i]; // SPD
        bs[i] = matrix::Random(dims[i], dim2s[i]);
        xs[i] = matrix(dims[i], dim2s[i]);
    }

    // Reference solve timing
    double ref_time = 0.0;

    timed_block_labeled("Eigen LLT", for (size_t i = 0; i < n_trials; i++) { x_refs[i] = As[i].llt().solve(bs[i]); });

    // BLASFEO solve timing
    double blasfeo_time = 0.0;
    timed_block_labeled("BLASFEO LLT", {
        utils::blasfeo_llt llt;
        for (size_t i = 0; i < n_trials; i++) {
            llt.compute(As[i]);
            llt.solve(bs[i], xs[i]);
        }
    });

    // Check correctness
    for (size_t i = 0; i < n_trials; i++) {
        REQUIRE((xs[i] - x_refs[i]).norm() < 1e-7);
    }
}