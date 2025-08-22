#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#include <thread>

#include <Eigen/LU>

void __attribute__((noinline)) dense_inner_product(const moto::matrix &rhs, moto::matrix &D, moto::matrix &out) {
    out.noalias() = rhs.transpose() * D * rhs;
}

moto::sparse_mat create_sp_inv(size_t nv) {
    using namespace moto;
    sparse_mat sp;
    size_t nj = nv - 6;
    scalar_t dt = 0.1;
    sp.insert<sparsity::eye>(0, 0, nv * 2);
    sp.insert<sparsity::diag>(0, nj, nj).setConstant(-dt);
    sp.insert(2 * nj, nj, 6, nv, sparsity::dense).setRandom();
    sp.insert(2 * nj + 6, nv + 6, 6, 6, sparsity::dense).setRandom();
    return sp;
}

moto::sparse_mat create_sp_f_x(size_t nv) {
    using namespace moto;
    sparse_mat sp;
    size_t nj = nv - 6;
    scalar_t dt = 0.1;
    sp.insert<sparsity::eye>(0, 0, 2 * nv);
    sp.insert(2 * nj, 0, 6, 2 * nv, sparsity::dense).setRandom();
    return sp;
}

moto::sparse_mat create_proj_f_u(size_t nv, size_t nf) {
    using namespace moto;
    sparse_mat sp;
    size_t nj = nv - 6;
    scalar_t dt = 0.1;
    sp.insert<sparsity::diag>(0, 0, nj).setConstant(dt * dt);
    sp.insert<sparsity::diag>(nj, 0, nj).setConstant(-dt);
    // sp.insert(2 * nj, nj, 12, nf, sparsity::dense).setRandom();
    sp.insert(2 * nj, 0, 12, nj + nf, sparsity::dense).setRandom();
    return sp;
}

moto::sparse_mat create_proj_f_x(size_t nv) {
    using namespace moto;
    sparse_mat sp;
    size_t nj = nv - 6;
    scalar_t dt = 0.1;
    sp.insert<sparsity::diag>(0, 0, nv + nj).setConstant(dt);
    sp.insert<sparsity::diag>(0, nj, nj).setConstant(-dt);
    // sp.insert(nv + nj, 0, 6, 2 * nv, sparsity::dense).setRandom();
    sp.insert(2 * nj, 0, 12, 2 * nv, sparsity::dense).setRandom();
    return sp;
}

TEST_CASE("inner_product") {
    using namespace moto;
    std::vector<sparse_mat> sp_inv, sp_f_x, proj_f_x;
    std::vector<matrix> D, dense_proj_f_x;
    size_t n_steps = 100;
    size_t n_trials = 100;

    size_t nv = 18;
    double total_dense_us = 0.0;
    double total_sparse_us = 0.0;
    size_t nf = 12;

    // fmt::println("proj_F_x \n{}", create_proj_f_u(nv, 12).dense().cast<bool>().cast<int>());
    // fmt::println("proj_F_x \n{}", create_proj_f_x(nv).dense().cast<bool>().cast<int>());
    for (size_t trial = 0; trial < n_trials; ++trial) {
        sp_inv.clear();
        sp_f_x.clear();
        proj_f_x.clear();
        D.clear();
        dense_proj_f_x.clear();

        for (size_t i = 0; i < n_steps; i++) {
            sp_inv.push_back(create_sp_inv(nv));
            sp_f_x.push_back(create_sp_f_x(nv));
            proj_f_x.push_back(create_proj_f_u(nv, nf));
            // proj_f_x.push_back(create_proj_f_x(nv));
            D.push_back(matrix::Random(2 * nv, 2 * nv));
            dense_proj_f_x.push_back(proj_f_x.back().dense());
        }

        matrix U = matrix::Zero(2 * nf, 2 * nf);
        // matrix U = matrix::Zero(2 * nv, 2 * nv);

        // Simulate cache miss by touching a large buffer
        {
            static std::vector<char> cache_miss_buffer(64 * 1024 * 1024, 1); // 64MB
            for (size_t i = 0; i < cache_miss_buffer.size(); i += 4096) {
                cache_miss_buffer[i]++;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        {
            auto start = std::chrono::high_resolution_clock::now();
            auto n = n_steps;
            while (n--) {
                dense_inner_product(dense_proj_f_x[n], D[n], U);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            total_dense_us += elapsed_us;
        }
        {
            static std::vector<char> cache_miss_buffer(64 * 1024 * 1024, 1); // 64MB
            for (size_t i = 0; i < cache_miss_buffer.size(); i += 4096) {
                cache_miss_buffer[i]++;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        {
            auto start = std::chrono::high_resolution_clock::now();
            auto n = n_steps;
            while (n--) {
                auto &jac_sp = proj_f_x[n];
                jac_sp.inner_product(D[n], U);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            total_sparse_us += elapsed_us;
        }
    }

    std::cout << "Average dense_inner_product: " << (total_dense_us / n_trials) << " us" << std::endl;
    std::cout << "Average sparse_inner_product: " << (total_sparse_us / n_trials) << " us" << std::endl;
}