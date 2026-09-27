#include <catch2/catch_test_macros.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_llt.hpp>

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
