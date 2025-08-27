#define CATCH_CONFIG_MAIN
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/utils/blasfeo_factorizer/blasfeo_llt.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_lu.hpp>
#include <moto/utils/blasfeo_factorizer/blasfeo_qr.hpp>

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

    // // Check correctness
    // for (size_t i = 0; i < n_trials; i++) {
    //     REQUIRE((xs[i] - x_refs[i]).norm() < 1e-5);
    // }
}

TEST_CASE("lu_test") {
    using namespace moto;
    size_t n_trials = 100;
    std::vector<matrix> As(n_trials), bs(n_trials), cs(n_trials), x_refs(n_trials), y_refs(n_trials), xs(n_trials), ys(n_trials);
    size_t nc = 37, nu = 25, nx = 37;
    // Generate random matrices
    for (size_t i = 0; i < n_trials; i++) {
        Eigen::MatrixXd temp = matrix::Random(nc, nx);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(temp);
        while (lu.rank() < std::min(nc, nx)) {
            temp = matrix::Random(nc, nx);
            lu.compute(temp);
        }
        As[i] = temp;
        bs[i] = matrix::Random(nc, nx);
        cs[i] = matrix::Random(nc, nu);
        xs[i] = matrix(nx, nx);
    }

    // Reference solve timing
    double ref_time = 0.0;

    Eigen::PartialPivLU<Eigen::MatrixXd> lu;

    timed_block_labeled("Eigen LU", for (size_t i = 0; i < n_trials; i++) { 
        lu.compute(As[i]);
        x_refs[i] = lu.solve(bs[i]); 
        // y_refs[i] = lu.solve(cs[i]);
    });

    // BLASFEO solve timing
    double blasfeo_time = 0.0;
    timed_block_labeled("BLASFEO LU", {
        utils::blasfeo_lu lu;
        for (size_t i = 0; i < n_trials; i++) {
            lu.compute(As[i]);
            lu.solve(bs[i], xs[i]);
            // lu.solve(cs[i], ys[i]);
        }
    });

    // Check correctness
    for (size_t i = 0; i < n_trials; i++) {
        REQUIRE((x_refs[i].allFinite() && !x_refs[i].hasNaN()));
        REQUIRE((xs[i] - x_refs[i]).norm() < 1e-5);
    }
}

TEST_CASE("BLASFEO LU transpose solve") {
    using namespace moto;
    size_t n_trials = 50;
    size_t nc = 37, nu = 15, nx = 37;
    std::vector<matrix> As(n_trials), bs(n_trials), x_refs(n_trials), xs(n_trials);

    // Generate random full-rank matrices
    for (size_t i = 0; i < n_trials; i++) {
        Eigen::MatrixXd temp = matrix::Random(nc, nx);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(temp);
        while (lu.rank() != nc) {
            temp.setRandom();
            lu.compute(temp);
        }
        As[i] = temp;
        bs[i] = matrix::Random(nx, nu);
        xs[i] = matrix(nx, nu);
    }

    Eigen::PartialPivLU<Eigen::MatrixXd> lu;

    timed_block_labeled("Eigen LU trsolve", for (size_t i = 0; i < n_trials; i++) { 
        lu.compute(As[i]);
        x_refs[i] = lu.solve(bs[i]); });

    // BLASFEO LU transpose solve
    utils::blasfeo_lu b_lu;
    timed_block_labeled("BLASFEO LU trsolve", {
        for (size_t i = 0; i < n_trials; i++) {
            b_lu.compute(As[i]);
            b_lu.transpose_solve(bs[i], xs[i]);
        }
    });

    // Check correctness
    for (size_t i = 0; i < n_trials; i++) {
        REQUIRE((As[i].transpose() * xs[i] - bs[i]).cwiseAbs().maxCoeff() < 1e-10);
    }
}

TEST_CASE("BLASFEO qr solve") {
    using namespace moto;
    size_t n_trials = 50;
    size_t nc = 12, nu = 25, nx = 37;
    std::vector<matrix> As(n_trials), bs(n_trials), x_refs(n_trials), xs(n_trials), QR(n_trials), QR_blasfeo(n_trials);

    // Generate random full-rank matrices
    for (size_t i = 0; i < n_trials; i++) {
        Eigen::MatrixXd temp = matrix::Random(nc, nu);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(temp);
        while (lu.rank() != nc) {
            temp.setRandom();
            lu.compute(temp);
        }
        As[i] = temp;
        bs[i] = matrix::Random(nc, nx);
        xs[i] = matrix(nu, nx);
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;

    timed_block_labeled("Eigen qr solve", {for (size_t i = 0; i < n_trials; i++) { 
        qr.compute(As[i]);
        // QR.emplace_back(qr.matrixQR());
        x_refs[i] = qr.solve(bs[i]); } });

    // BLASFEO LU transpose solve
    utils::blasfeo_qr b_qr;
    std::vector<double> res(n_trials, 0.);
    timed_block_labeled("BLASFEO qr solve", {
        for (size_t i = 0; i < n_trials; i++) {
            b_qr.compute(As[i]);
            // QR_blasfeo.emplace_back(matrix());
            // b_qr.LQ_.to_eigen(QR_blasfeo.back());
            // fmt::print("BLASFEO QR {}x{}:\n{}\n", QR_blasfeo.back().rows(), QR_blasfeo.back().cols(), QR_blasfeo.back());
            // fmt::print("Eigen QR {}x{}:\n{}\n", QR[i].rows(), QR[i].cols(), QR[i]);
            b_qr.solve(bs[i], xs[i]);
            res.emplace_back((As[i] * xs[i] - bs[i]).cwiseAbs().maxCoeff());
            // fmt::print("Residual {}x{} max {}:\n{}\n", res.rows(), res.cols(), res.cwiseAbs().maxCoeff(), res);
            // fmt::print("BLASFEO x {}x{}:\n{}\n", xs[i].rows(), xs[i].cols(), xs[i]);
            // fmt::print("Eigen x {}x{}:\n{}\n", x_refs[i].rows(), x_refs[i].cols(), x_refs[i]);
        }
    });

    // Check correctness
    for (size_t i = 0; i < n_trials; i++) {
        REQUIRE(res[i] < 1e-10);
    }
}

// TEST_CASE("BLASFEO LQ redundance") {
//     using namespace moto;
//     size_t n_trials = 50;
//     size_t nc = 6, nu = 25, nx = 37;
//     matrix A = matrix::Zero(nc, nu);
//     A.bottomRightCorner(3, 3).setIdentity();
//     fmt::print("A {}x{}:\n{}\n", A.rows(), A.cols(), A);
//     matrix LQ;
//     moto::utils::blasfeo_qr b_qr;
//     b_qr.compute(A);
//     b_qr.LQ_.to_eigen(LQ);
//     fmt::print("LQ {}x{}:\n{}\n", LQ.rows(), LQ.cols(), LQ);
//     matrix Q;
//     b_qr.Q(Q);
//     fmt::print("Q {}x{}:\n{}\n", Q.rows(), Q.cols(), Q);
//     fmt::print("Q^T * Q:\n{}\n", Q.transpose() * Q);
//     matrix L;
//     b_qr.L(L);
//     fmt::print("L {}x{}:\n{}\n", L.rows(), L.cols(), L);
//     matrix res;
//     b_qr.residual(res);
//     fmt::print("Residual {}x{} max {}:\n{}\n", res.rows(), res.cols(), res.cwiseAbs().maxCoeff(), res);
// }