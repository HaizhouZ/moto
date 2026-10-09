#include <catch2/catch_test_macros.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_fullpiv_lu.hpp>
#include <Eigen/LU>
#include <array>
#include <bit>
#include <random>
#include <thread>

using namespace moto;

namespace {
matrix random_matrix(int m, int n, std::mt19937_64 &rng) {
    std::normal_distribution<double> normal;
    matrix a(m, n);
    for (auto &value : a.reshaped()) value = normal(rng);
    return a;
}

void check_factor(utils::blasfeo_fullpiv_lu &factor, const matrix &a, std::mt19937_64 &rng) {
    const matrix original = a;
    factor.compute(a);
    REQUIRE((a.array() == original.array()).all());
    const double scale = std::max(1e-300, a.norm());
    REQUIRE((factor.reconstruction() - a).norm() < 2e-12 * scale);
    const matrix z = factor.kernel();
    REQUIRE(z.rows() == a.cols());
    REQUIRE(z.cols() == a.cols() - factor.rank());
    REQUIRE(z.allFinite());
    REQUIRE((a * z).norm() < 2e-12 * scale * std::max(1., z.norm()));
    const matrix b = a * random_matrix(a.cols(), 3, rng);
    const matrix c = a.transpose() * random_matrix(a.rows(), 3, rng);
    matrix x(a.cols(), 3), y(a.rows(), 3);
    factor.solve(b, x);
    factor.transpose_solve(c, y);
    REQUIRE(x.allFinite());
    REQUIRE(y.allFinite());
    REQUIRE((a * x - b).norm() < 2e-12 * std::max(1e-300, scale*x.norm() + b.norm()));
    REQUIRE((a.transpose() * y - c).norm() < 2e-12 * std::max(1e-300, scale*y.norm() + c.norm()));
    Eigen::FullPivLU<matrix> reference(a);
    REQUIRE(factor.rank() == reference.rank());
    // Native builds align update arithmetic. Other Eigen compiler policies may
    // choose different subthreshold factors; residual identities above apply.
    if (reference.rank() == std::min(a.rows(), a.cols())) {
        const matrix expected_x = reference.solve(b), expected_y = reference.transpose().solve(c);
        REQUIRE((x - expected_x).norm() < 2e-9 * std::max(1., expected_x.norm()));
        REQUIRE((y - expected_y).norm() < 2e-9 * std::max(1., expected_y.norm()));
    }
}
}

TEST_CASE("Panel LU preserves rectangular geometry across workspace reuse", "[factorization][fullpiv]") {
    std::mt19937_64 rng(1020261010);
    utils::blasfeo_fullpiv_lu factor;
    for (int trial = 0; trial < 240; ++trial) {
        const int m = 1 + trial % 32, n = 1 + trial * 17 % 35;
        CAPTURE(trial, m, n);
        matrix a;
        switch (trial % 4) {
        case 0: a = random_matrix(m, n, rng); break;
        case 1:
            a = random_matrix(m, n, rng);
            if (m > 1) a.row(m-1) = a.row(0);
            break;
        case 2: {
            const int r = std::min(m,n) / 2;
            a = random_matrix(m,r,rng) * random_matrix(r,n,rng);
            break;
        }
        default: a = matrix::Zero(m,n); break;
        }
        a *= trial % 3 == 0 ? 1e-20 : trial % 3 == 1 ? 1. : 1e20;
        check_factor(factor, a, rng);
    }
}

TEST_CASE("Panel LU selects actual nonprefix pivots for its nullspace", "[factorization][fullpiv]") {
    utils::blasfeo_fullpiv_lu factor;
    for (int block : {4, 8, 16}) for (int leading : {1, 2, 3})
    for (int padding : {0, 3, 7}) for (double scale : {1e-30, 1., 1e30}) {
        const int m = leading + block, n = m + padding;
        matrix a = matrix::Zero(m,n);
        for (int i=0; i<leading; ++i) a(i,i) = 1. - .125*i;
        const double tiny = std::numeric_limits<double>::epsilon() * m * .6;
        for (int i=0; i<block; ++i) for (int j=0; j<block; ++j)
            a(leading+i,leading+j) = std::popcount(static_cast<unsigned>(i&j)) % 2 ? -tiny : tiny;
        a *= scale;
        factor.compute(a);
        const matrix z = factor.kernel();
        REQUIRE(z.cols() == n - factor.rank());
        REQUIRE(z.allFinite());
        REQUIRE(z.norm() < 100.);
        REQUIRE((a*z).norm() < 2e-12*a.norm()*z.norm());
        REQUIRE((z.transpose()*z).fullPivLu().rank() == z.cols());
    }
}

TEST_CASE("Panel pivot reduction preserves ties and masks reused panel tails", "[factorization][fullpiv]") {
    utils::blasfeo_fullpiv_lu factor;
    for (int m : {1,3,4,5,7,8,9}) for (int n : {2,5,9,17}) {
        CAPTURE(m,n);
        // Equal-magnitude pivots in different panels and columns must still
        // choose the first column, then the first row, including redundant rows.
        matrix a=matrix::Zero(m,n);
        a.col(1).setOnes();
        a.col(n-1).setConstant(-1.);
        factor.compute(a);
        Eigen::FullPivLU<matrix> reference(a);
        REQUIRE(factor.rank()==reference.rank());
        matrix b=matrix::Ones(m,2), c=matrix::Ones(n,2);
        matrix x(n,2), y(m,2);
        factor.solve(b,x); factor.transpose_solve(c,y);
        REQUIRE(x.isApprox(reference.solve(b),1e-12));
        REQUIRE(y.isApprox(reference.transpose().solve(c),1e-12));
        const matrix z=factor.kernel();
        REQUIRE((a*z).norm()<1e-12);
    }
    matrix overflow=matrix::Constant(8,8,std::numeric_limits<double>::max());
    overflow(7,7)=-std::numeric_limits<double>::max();
    REQUIRE_THROWS_AS(factor.compute(overflow),std::runtime_error);
    // The larger invalid factor leaves nonfinite entries outside the next
    // active shape. Tail padding must not enter its magnitude reduction.
    matrix small=matrix::Ones(1,2);
    factor.compute(small);
    REQUIRE(factor.rank()==1);
    REQUIRE((small*factor.kernel()).norm()<1e-12);
}

TEST_CASE("Panel LU handles empty geometry, move ownership and invalid inputs", "[factorization][fullpiv]") {
    utils::blasfeo_fullpiv_lu factor;
    REQUIRE_THROWS_AS(factor.rank(), std::logic_error);
    for (auto shape : {std::pair{0,0}, {0,7}, {5,0}}) {
        matrix a(shape.first,shape.second);
        factor.compute(a);
        REQUIRE(factor.rank() == 0);
        REQUIRE(factor.kernel().rows() == shape.second);
        REQUIRE(factor.kernel().cols() == shape.second);
        matrix b = matrix::Zero(shape.first,2), x(shape.second,2);
        factor.solve(b,x);
        REQUIRE(x.isZero());
    }
    factor.compute(matrix::Identity(3,3));
    auto moved = std::move(factor);
    vector b = vector::Ones(3), x(3);
    moved.solve(b,x);
    REQUIRE(x.isApprox(b));
    matrix bad = matrix::Identity(3,3);
    bad(0,0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(moved.compute(bad), std::invalid_argument);
    bad(0,0) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(moved.compute(bad), std::invalid_argument);
    vector wrong(2);
    REQUIRE_THROWS_AS(moved.solve(b,wrong), std::invalid_argument);
    const double big = std::numeric_limits<double>::max();
    matrix overflow(2,2); overflow << big,big,big,-big;
    REQUIRE_THROWS_AS(moved.compute(overflow), std::runtime_error);
    REQUIRE_THROWS_AS(moved.kernel(), std::logic_error);
    moved.compute(matrix::Identity(3,3));
    moved.solve(b,x);
    REQUIRE(x.isApprox(b));
}

TEST_CASE("Panel LU workspaces are private to each worker", "[factorization][fullpiv]") {
    std::array<bool,6> passed{};
    std::vector<std::thread> workers;
    for (size_t worker=0; worker<passed.size(); ++worker) workers.emplace_back([&,worker] {
        utils::blasfeo_fullpiv_lu factor;
        std::mt19937_64 rng(620261010+worker);
        bool valid = true;
        for (int trial=0; trial<100; ++trial) {
            const int m=1+trial%24, n=1+trial*7%29;
            const matrix a=random_matrix(m,n,rng);
            factor.compute(a);
            const matrix z=factor.kernel();
            valid = valid && z.allFinite() &&
                (a*z).norm() < 2e-12*a.norm()*std::max(1.,z.norm());
        }
        passed[worker]=valid;
    });
    for (auto &worker : workers) worker.join();
    for (bool valid : passed) REQUIRE(valid);
}
