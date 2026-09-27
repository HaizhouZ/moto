#include <catch2/catch_test_macros.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_llt.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lq.hpp>

#include <chrono>
#include <iostream>

using namespace moto;

TEST_CASE("BLASFEO Cholesky rejects zero and negative pivots", "[factorization]") {
    utils::blasfeo_llt factor;
    matrix a = matrix::Identity(3, 3);
    SECTION("singular positive semidefinite matrix") {
        a(1, 1) = 0.;
    }
    SECTION("negative diagonal") {
        a(1, 1) = -1.;
    }
    SECTION("indefinite with positive diagonal") {
        a(0, 1) = a(1, 0) = 2.;
    }
    factor.compute(a);
    REQUIRE_FALSE(factor.valid());
}

TEST_CASE("BLASFEO Cholesky accepts SPD matrices and solves accurately", "[factorization]") {
    matrix a(3, 3);
    a << 4., 1., .5, 1., 3., .2, .5, .2, 2.;
    utils::blasfeo_llt factor;
    factor.compute(a);
    REQUIRE(factor.valid());
    matrix b = matrix::Identity(3, 3), x(3, 3);
    factor.solve(b, x);
    REQUIRE((a * x - b).norm() < 1e-12);
    vector rhs = vector::Ones(3), result(3);
    factor.solve(rhs, result);
    REQUIRE((a * result - rhs).norm() < 1e-12);
}

TEST_CASE("BLASFEO LQ supplies the hard-equality projection geometry",
          "[factorization][lq]") {
    matrix a = matrix::Random(7, 19);
    matrix rhs = matrix::Random(7, 11);
    vector transpose_solution = vector::Random(7);
    vector transpose_rhs = a.transpose() * transpose_solution;

    utils::blasfeo_lq factor;
    factor.compute(a);
    REQUIRE(factor.uses_blasfeo());
    REQUIRE(factor.rank() == 7);

    matrix solution;
    factor.solve(rhs, solution);
    REQUIRE((a * solution - rhs).norm() < 1e-11);

    matrix kernel;
    factor.kernel(kernel);
    REQUIRE(kernel.rows() == 19);
    REQUIRE(kernel.cols() == 12);
    REQUIRE((a * kernel).norm() < 1e-11);
    REQUIRE((kernel.transpose() * kernel - matrix::Identity(12, 12)).norm() <
            1e-11);

    vector recovered;
    factor.transpose_solve(transpose_rhs, recovered);
    REQUIRE((recovered - transpose_solution).norm() < 1e-11);
}

TEST_CASE("BLASFEO LQ preserves rank-deficient projection behavior",
          "[factorization][lq]") {
    matrix a = matrix::Random(6, 13);
    a.row(5) = a.row(2);
    vector rhs = a * vector::Random(13);

    utils::blasfeo_lq factor;
    factor.compute(a);
    REQUIRE_FALSE(factor.uses_blasfeo());
    REQUIRE(factor.rank() == 5);

    vector solution;
    factor.solve(rhs, solution);
    REQUIRE((a * solution - rhs).norm() < 1e-11);

    matrix kernel;
    factor.kernel(kernel);
    REQUIRE(kernel.cols() == 8);
    REQUIRE((a * kernel).norm() < 1e-11);
}

TEST_CASE("benchmark hard-equality projection factorization",
          "[.benchmark][factorization][lq]") {
    using clock = std::chrono::steady_clock;
    constexpr size_t repetitions = 2000;
    for (const auto [rows, cols] :
         {std::pair<Eigen::Index, Eigen::Index>{6, 24}, {12, 48},
          {24, 96}, {48, 96}, {70, 96}}) {
        matrix a = matrix::Random(rows, cols);
        matrix rhs = matrix::Random(rows, cols);
        vector vector_rhs = matrix::Random(rows, 1);
        vector multiplier = matrix::Random(rows, 1);
        vector transpose_rhs = a.transpose() * multiplier;
        matrix kernel, solution;
        vector vector_solution, multiplier_solution;
        volatile double checksum = 0.;

        Eigen::FullPivLU<matrix> eigen;
        utils::blasfeo_lq blasfeo;
        for (size_t i = 0; i < 20; ++i) {
            eigen.compute(a);
            kernel = eigen.kernel();
            blasfeo.compute(a);
            blasfeo.kernel(kernel);
        }

        const auto eigen_factor_begin = clock::now();
        for (size_t i = 0; i < repetitions; ++i) {
            eigen.compute(a);
            kernel = eigen.kernel();
            checksum = checksum + kernel(0, 0);
        }
        const auto eigen_factor_end = clock::now();

        const auto blasfeo_factor_begin = clock::now();
        for (size_t i = 0; i < repetitions; ++i) {
            blasfeo.compute(a);
            blasfeo.kernel(kernel);
            checksum = checksum + kernel(0, 0);
        }
        const auto blasfeo_factor_end = clock::now();

        const auto eigen_projection_begin = clock::now();
        for (size_t i = 0; i < repetitions; ++i) {
            eigen.compute(a);
            kernel = eigen.kernel();
            solution = eigen.solve(rhs);
            vector_solution = eigen.solve(vector_rhs);
            multiplier_solution = eigen.transpose().solve(transpose_rhs);
            checksum = checksum + solution(0, 0) + vector_solution(0) +
                       multiplier_solution(0);
        }
        const auto eigen_projection_end = clock::now();

        const auto blasfeo_projection_begin = clock::now();
        for (size_t i = 0; i < repetitions; ++i) {
            blasfeo.compute(a);
            blasfeo.kernel(kernel);
            blasfeo.solve(rhs, solution);
            blasfeo.solve(vector_rhs, vector_solution);
            blasfeo.transpose_solve(transpose_rhs, multiplier_solution);
            checksum = checksum + solution(0, 0) + vector_solution(0) +
                       multiplier_solution(0);
        }
        const auto blasfeo_projection_end = clock::now();

        const auto us = [](auto begin, auto end) {
            return std::chrono::duration<double, std::micro>(end - begin)
                       .count() /
                   static_cast<double>(repetitions);
        };
        const double eigen_factor = us(eigen_factor_begin, eigen_factor_end);
        const double blasfeo_factor =
            us(blasfeo_factor_begin, blasfeo_factor_end);
        const double eigen_projection =
            us(eigen_projection_begin, eigen_projection_end);
        const double blasfeo_projection =
            us(blasfeo_projection_begin, blasfeo_projection_end);
        std::cout << rows << "x" << cols << " factor+kernel: Eigen "
                  << eigen_factor << " us, BLASFEO " << blasfeo_factor
                  << " us, speedup " << eigen_factor / blasfeo_factor
                  << "x; full projection: Eigen " << eigen_projection
                  << " us, BLASFEO " << blasfeo_projection << " us, speedup "
                  << eigen_projection / blasfeo_projection
                  << "x; checksum " << checksum << '\n';
    }
}
