#include <iostream>
#include <random>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

#include <thread>

#include <moto/spmm/sparse_mat.hpp>

namespace moto {

auto generate_random_sparse_mat() {
    std::mt19937 rng0(std::random_device{}());
    std::uniform_int_distribution<size_t> rows_dist(10, 80);
    std::uniform_int_distribution<size_t> cols_dist(10, 80);
    std::uniform_int_distribution<size_t> panels_dist(1, 6);
    size_t rows = rows_dist(rng0), cols = cols_dist(rng0), num_panels = panels_dist(rng0);
    sparse_mat sm;
    std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> sp_dist({1, 10, 10});
    std::uniform_int_distribution<int> sz_dist(1, std::min(rows, cols) / 2 + 1);
    std::uniform_int_distribution<int> pos_dist_row(0, rows - 1);
    std::uniform_int_distribution<int> pos_dist_col(0, cols - 1);

    for (int i = 0; i < num_panels; ++i) {
        sparsity sp = static_cast<sparsity>(sp_dist(rng));
        int r = sz_dist(rng);
        int c = (sp == sparsity::diag || sp == sparsity::eye) ? r : sz_dist(rng);
        int r_st = pos_dist_row(rng) % (rows - r + 1);
        int c_st = pos_dist_col(rng) % (cols - c + 1);
        auto mat_ref = sm.insert(r_st, c_st, r, c, sp);
        if (sp == sparsity::dense || sp == sparsity::diag) {
            mat_ref.setRandom();
        }
    }
    return sm;
};

// Accepts a member function pointer for sparse_mat inner product
template <typename D_type, typename out_type, typename callback>
    requires std::is_invocable_v<callback, const D_type &, matrix &>
void test_unary(std::string_view name, void (sparse_mat::*func)(const D_type &, out_type &), callback &&ground_truth) {

    scalar_t max_err = 0;
    scalar_t sum_err = 0;
    size_t ntrials = 100;
    for (int test_idx = 0; test_idx < ntrials; ++test_idx) {
        auto sm = generate_random_sparse_mat();
        auto dense = sm.dense();
        matrix V = matrix::Random(sm.rows_, sm.rows_);
        V = V * V.transpose();
        matrix B = matrix::Zero(sm.cols_, sm.cols_);
        matrix B_dense = matrix::Zero(sm.cols_, sm.cols_);
        (sm.*func)(V, B);
        B_dense.noalias() = ground_truth(V, dense);
        max_err = std::max(max_err, (B - B_dense).cwiseAbs().maxCoeff());
        sum_err = std::max(sum_err, (B - B_dense).cwiseAbs().sum());
    }
    std::cout << "Random sparse_mat [" << name << "] accuracy check:\n";
    std::cout << "Max abs diff: " << max_err << std::endl;
    std::cout << "Sum abs diff: " << sum_err << std::endl;
    REQUIRE(max_err <= 1e-10);
    REQUIRE(sum_err <= 1e-10);
}

template <typename other_t>
std::pair<size_t, size_t> out_dim(std::string_view name, sparse_mat &sm, other_t &others) {
    if (name == "times") {
        return {sm.rows_, others.cols()};
    } else if (name == "T_times") {
        return {sm.cols_, others.cols()};
    } else if (name == "right_times") {
        return {others.rows(), sm.cols_};
    } else if (name == "right_T_times") {
        return {others.cols(), sm.cols_};
    } else {
        throw std::runtime_error("Unknown operation: " + std::string(name));
    }
};

size_t _inner_dim(std::string_view name, sparse_mat &sm) {
    if (name == "times") {
        return sm.cols_;
    } else if (name == "T_times") {
        return sm.rows_;
    } else if (name == "right_times") {
        return sm.rows_;
    } else if (name == "right_T_times") {
        return sm.rows_;
    } else {
        throw std::runtime_error("Unknown operation: " + std::string(name));
    }
};

template <typename other_t, typename out_type>
other_t make_other(std::string_view name, sparse_mat &sm) {
    constexpr bool row_fixed = out_type::RowsAtCompileTime != Eigen::Dynamic;
    constexpr size_t row_ = row_fixed ? out_type::RowsAtCompileTime : 0;
    constexpr bool col_fixed = out_type::ColsAtCompileTime != Eigen::Dynamic;
    constexpr size_t col_ = col_fixed ? out_type::ColsAtCompileTime : 0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(1, 30);
    size_t outer_dim = dist(rng);
    size_t inner_dim = _inner_dim(name, sm);
    if (name == "times") {
        return other_t::Random(inner_dim, col_fixed ? col_ : outer_dim);
    } else if (name == "T_times") {
        return other_t::Random(inner_dim, col_fixed ? col_ : outer_dim);
    } else if (name == "right_times") {
        return other_t::Random(row_fixed ? row_ : outer_dim, inner_dim);
    } else if (name == "right_T_times") {
        return other_t::Random(inner_dim, row_fixed ? row_ : outer_dim);
    } else {
        throw std::runtime_error("Unknown operation: " + std::string(name));
    }
}

// Accepts a member function pointer for sparse_mat inner product
template <spmm::binary_op_type op, bool add, typename D_type, typename out_type, typename callback>
// requires std::is_invocable_v<callback, const D_type &, const matrix &>
void test_binary(std::string_view name,
                 void (sparse_mat::*func)(const D_type &, out_type &),
                 callback &&ground_truth) {
    if constexpr (!spmm::is_consistent<D_type, out_type>::value(op))
        return;

    scalar_t max_err = 0;
    scalar_t sum_err = 0;
    size_t ntrials = 100;
    for (int test_idx = 0; test_idx < ntrials; ++test_idx) {
        auto sm = generate_random_sparse_mat();
        size_t inner_dim = _inner_dim(name, sm);
        auto dense = sm.dense();
        auto V = make_other<D_type, out_type>(name, sm);
        auto [out_rows, out_cols] = out_dim(name, sm, V);
        out_type B = out_type::Zero(out_rows, out_cols);
        out_type B_dense = out_type::Zero(out_rows, out_cols);
        (sm.*func)(V, B);
        B_dense.noalias() = ground_truth(V, dense, std::bool_constant<add>{});
        max_err = std::max(max_err, (B - B_dense).cwiseAbs().maxCoeff());
        sum_err = std::max(sum_err, (B - B_dense).cwiseAbs().sum());
    }
    REQUIRE(max_err <= 1e-12);
    REQUIRE(sum_err <= 1e-12);
    std::cout << "Random sparse_mat [" << name << ", " << add << "] accuracy check:\n";
    std::cout << "Max abs diff: " << max_err << std::endl;
    std::cout << "Sum abs diff: " << sum_err << std::endl;
}

// void test_speed(size_t nv) {
//     sparse_mat euler;
//     // set pos r = r + v * dt
//     euler.insert(0, 0, 2 * nv, 2 * nv, sparsity::eye);
//     euler.insert(0, nv, nv, nv, sparsity::diag);
//     // set vel v = v + a * dt
//     // euler.insert(nv, nv, nv, nv, sparsity::eye);

//     std::cout << "euler matrix:\n"
//               << euler.dense() << std::endl;

//     sparse_mat lie_euler;

//     // lie_euler.insert(0, 0, 4, 3, sparsity::dense);         // for quaternion
//     // lie_euler.insert(4, 3, nv - 3, nv - 3, sparsity::eye); // for position
//     lie_euler.insert(0, 0, 2 * nv + 1, 2 * nv + 1, sparsity::eye);
//     lie_euler.insert(3, nv + 3, 4, 3, sparsity::dense);
//     lie_euler.insert(0, nv + 1, nv, nv, sparsity::diag);
//     lie_euler.rows_ = 2 * nv + 1;
//     // set vel v = v + a * dt
//     // lie_euler.insert(nv + 1, nv, nv, nv, sparsity::eye);

//     std::cout << "lie_euler matrix:\n"
//               << lie_euler.dense() << std::endl;

//     sparse_mat inv_euler;
//     // inv_euler.insert(0, 0, 4, 3, sparsity::dense);         // for quaternion
//     // inv_euler.insert(4, 3, nv - 3, nv - 3, sparsity::eye); // for position
//     inv_euler.insert(0, 0, nv + 1, nv + 1, sparsity::eye);
//     // inv_euler.insert(3, nv + 1 + 3, 4, 3, sparsity::dense);
//     // inv_euler.insert(0, nv + 1, 3, 3, sparsity::eye);
//     // inv_euler.insert(0, nv + 1, nv, nv, sparsity::eye);
//     inv_euler.insert(3, nv + 3, 4, 3, sparsity::dense).setOnes();
//     inv_euler.insert(0, nv + 1, nv, nv, sparsity::diag).setOnes();
//     // inv_euler.insert(7, nv + 1 + 3 + 3, nv - 6, nv - 6, sparsity::eye);
//     // set rnea
//     // inv_euler.insert(nv + 1, nv + 1, nv, nv, sparsity::dense);
//     inv_euler.insert(nv + 1, 0, nv, 2 * nv + 1, sparsity::dense).setOnes();

//     std::cout << "inv_euler matrix:\n"
//               << inv_euler.dense() << std::endl;

//     size_t ntrials = 10000;

//     std::vector<std::pair<std::string, sparse_mat *>> mats = {
//         // {"euler", &euler},
//         // {"lie_euler", &lie_euler},
//         {"inv_euler", &inv_euler}

//     };
//     size_t n_runs = 0;

//     for (size_t trial = 0; trial < ntrials; ++trial) {

//         for (auto &[name, sm] : mats) {
//             // Ensure B has correct size for each sparse_mat
//             matrix B = matrix::Random(sm->rows_, sm->rows_);
//             B = B * B.transpose(); // make B symmetric
//             matrix out = matrix::Zero(sm->cols_, sm->cols_);
//             auto dense = sm->dense();
//             matrix out_ref = matrix::Zero(sm->cols_, sm->cols_);
//             {
//                 timed_block_labeled("dense", out_ref.noalias() = dense.transpose() * B * dense);
//             }
//             {
//                 timed_block_labeled("sparse", sm->inner_product(B, out));
//             }
//             // std::cout << name << " trial " << trial << " max abs diff: " << (out - out_ref).cwiseAbs().maxCoeff() << std::endl;
//         }
//     }
//     // std::cout << "Average number of runs per multiplication: " << n_runs << std::endl;
//     moto::utils::timing_storage<"sparse">::count() /= 100;
//     moto::utils::timing_storage<"dense">::count() /= 100;
// }

} // namespace moto

#define TEST_UNARY(func, ground_truth) \
    test_unary(#func, &moto::sparse_mat::func, ground_truth)

#define TEST_BINARY_IMPL(func, lhs_type, rhs_type, ground_truth)                                                                                                                          \
    test_binary<moto::spmm::func, false>(#func, static_cast<void (moto::sparse_mat::*)(const lhs_type &, rhs_type &)>(&moto::sparse_mat::func<false, lhs_type, rhs_type>), ground_truth); \
    test_binary<moto::spmm::func, true>(#func, static_cast<void (moto::sparse_mat::*)(const lhs_type &, rhs_type &)>(&moto::sparse_mat::func<true, lhs_type, rhs_type>), ground_truth);
  // TEST_BINARY_IMPL(func, sparse_mat, matrix, ground_truth);     \
    // TEST_BINARY_IMPL(func, sparse_mat, vector, ground_truth);     \

#define TEST_BINARY(func, ground_truth)                           \
    TEST_BINARY_IMPL(func, matrix, matrix, ground_truth);         \
    TEST_BINARY_IMPL(func, matrix, vector, ground_truth);         \
    TEST_BINARY_IMPL(func, row_vector, row_vector, ground_truth); \
    TEST_BINARY_IMPL(func, row_vector, vector, ground_truth);     \
    TEST_BINARY_IMPL(func, vector, row_vector, ground_truth);     \
    TEST_BINARY_IMPL(func, vector, vector, ground_truth);

TEST_CASE("SPARSE_MAT_OPS") {
    using namespace moto;
    TEST_UNARY(inner_product, [](const auto &V, const auto &dense) {
        return dense.transpose() * V * dense;
    });
    TEST_BINARY(times, []<bool add>(const auto &V, const matrix &dense, std::bool_constant<add> = {}) {
        if constexpr (add)
            return dense * V;
        else
            return -dense * V;
    });
    TEST_BINARY(T_times, []<bool add>(const auto &V, const matrix &dense, std::bool_constant<add> = {}) {
        if constexpr (add)
            return dense.transpose() * V;
        else
            return -dense.transpose() * V;
    });
    TEST_BINARY(right_times, []<bool add>(const auto &V, const matrix &dense, std::bool_constant<add> = {}) {
        if constexpr (add)
            return V * dense;
        else
            return -V * dense;
    });
    TEST_BINARY(right_T_times, []<bool add>(const auto &V, const matrix &dense, std::bool_constant<add> = {}) {
        if constexpr (add)
            return V.transpose() * dense;
        else
            return -V.transpose() * dense;
    });
    // moto::test_speed(18);
}