#define MOTO_NS_RICCATI_IMPL

#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/lbfgs.hpp>
#include <moto/utils/field_conversion.hpp>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
extern void kkt_diagnosis(ns_riccati_data *cur);
extern void print_debug(ns_riccati_data *cur);
void generic_solver::riccati_recursion(ns_riccati_data *cur, ns_riccati_data *prev) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    // check positiveness
    // bool qyy_invertible = isPositiveDefinite(d.V_yy);
    timed_block_start("V_yy symmetrize");
    d.V_yy.array() /= 2;
    /// @todo: temporary
    d.V_yy = d.V_yy + d.V_yy.transpose().eval();
    // d.V_yy.diagonal().array() += 10.0;
    timed_block_end("V_yy symmetrize");
    // d.V_yy.diagonal().array() += 1e-6; // ensure positive definiteness
    if (d.V_yy.hasNaN() || !d.V_yy.allFinite()) {
        fmt::print("V_yy: \n {}\n", d.V_yy);
        print_debug(cur);
        throw std::runtime_error("V_yy has NaN or inf");
    }
    // if (d.V_yy.llt().info() != Eigen::Success) {
    //     fmt::print("V_yy: \n {}\n", d.V_yy);
    //     // print eigenvalues
    //     auto eigs = d.V_yy.eigenvalues();
    //     fmt::print("V_yy eigenvalues: {}\n", eigs.transpose());
    //     print_debug(cur);
    //     throw std::runtime_error("V_yy is not positive definite");
    // }
    timed_block_start("compute_U");
    nsp.y_0_p_k.noalias() = d.Q_y.transpose() - d.V_yy * nsp.y_y_k;
    if (d.rank_status_ == rank_status::unconstrained) {
        d.F_x.right_times<false>(d.V_yy, nsp.y_0_p_K);
        d.F_u.inner_product(d.V_yy, nsp.Q_zz);
        // Structured L-BFGS additive correction A_k (unknown Hessian estimate).
        // Q_zz already holds C_k + H_IPM_k + F_u^T·V_yy·F_u at this point.
        // A_k is built from γ·I using stored (s, y#) pairs and added in.
        solver::lbfgs_apply_structured_to_qzz(nsp.Q_zz, d.lbfgs_);
        d.F_u.T_times<false>(nsp.y_0_p_k, nsp.z_0_k);
        d.F_u.T_times<false>(nsp.y_0_p_K, nsp.z_0_K);
    } else {
        nsp.y_0_p_K.noalias() -= d.V_yy * nsp.y_y_K;
        if (d.rank_status_ == rank_status::constrained) [[likely]] {
            nsp.Q_zz.noalias() += nsp.Z_y.transpose() * d.V_yy * nsp.Z_y; /// todo unconstrained
            // Structured L-BFGS additive correction A_k in null-space coordinates.
            // Q_zz already holds Z_u^T·(C_k+H_IPM_k)·Z_u + Z_y^T·V_yy·Z_y at this point.
            solver::lbfgs_apply_structured_to_qzz(nsp.Q_zz, nsp.Z_u, d.lbfgs_);
            // compute bar{y}_0
            nsp.z_0_k.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_k;
            nsp.z_0_K.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_K;
        }
    }
    timed_block_end("compute_U");
    // auto min_Q_uu = d.Q_uu.eigenvalues().real().minCoeff();
    // auto min_Q_yy = d.V_yy.eigenvalues().real().minCoeff();
    // auto min_Q_zz = nsp.Q_zz.eigenvalues().real().minCoeff();
    // if (min_Q_uu <= 0 || min_Q_yy <= 0 || min_Q_zz <= 0) {
    //     fmt::print("Q_uu min eigenvalue: {}\n", min_Q_uu);
    //     fmt::print("V_yy min eigenvalue: {}\n", min_Q_yy);
    //     fmt::print("Q_zz min eigenvalue: {}\n", min_Q_zz);
    //     throw std::runtime_error("One or more matrices are not positive definite");
    // }
    // fmt::print("u_0_p_k: \n{}\n", nsp.u_0_p_k.transpose());
    // fmt::print("u_0_p_K: \n{}\n", nsp.u_0_p_K.transpose());
    // fmt::print("u_y_K: \n{}\n", nsp.u_y_K.transpose());
    // compute z_u
    if (d.rank_status_ == rank_status::unconstrained) {
        timed_block_start("solve_nullspace");
        // nsp.z_K = -nsp.z_0_K;
        nsp.llt_ns_.compute(nsp.Q_zz);
        // if (nsp.llt_ns_.info() != Eigen::Success) {
        if (!nsp.llt_ns_.valid()) {
            fmt::print("Q_uu: \n{}\n", d.Q_uu.dense());
            fmt::print("V_yy: \n{}\n", d.V_yy);
            fmt::print("Q_zz: \n{}\n", nsp.Q_zz);
            kkt_diagnosis(cur);
            print_debug(cur);
            throw std::runtime_error("U is not positive definite");
        } else {
            nsp.llt_ns_.solve(nsp.z_0_K, nsp.z_K, -1.0);
        }
        // nsp.llt_ns_.solveInPlace(nsp.z_K);
        timed_block_end("solve_nullspace");
    } else {
        // generic_constr rank > 0
        if (d.rank_status_ == rank_status::fully_constrained) {
            // fully constrained
        } else [[likely]] {
            // solve nullspace system
            // nsp.z_K = -nsp.z_0_K;
            timed_block_start("solve_nullspace");
            nsp.llt_ns_.compute(nsp.Q_zz);
            // nsp.llt_ns_.solveInPlace(nsp.z_K);
            nsp.llt_ns_.solve(nsp.z_0_K, nsp.z_K, -1.0);
            timed_block_end("solve_nullspace");
        }
    }
    timed_block_start("update_value_function");
    if (d.rank_status_ == rank_status::fully_constrained) {
        d.Q_x.noalias() += -nsp.y_0_p_k.transpose() * nsp.y_y_K;
        d.V_xx.noalias() += -nsp.y_0_p_K.transpose() * nsp.y_y_K;
    } else if (d.rank_status_ == rank_status::constrained) [[likely]] {
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K - nsp.y_0_p_k.transpose() * nsp.y_y_K;
        d.V_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K - nsp.y_0_p_K.transpose() * nsp.y_y_K;
    } else {
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K;
        d.F_x.right_T_times<false>(nsp.y_0_p_k, d.Q_x);
        d.V_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K;
        d.F_x.right_T_times<false>(nsp.y_0_p_K, d.V_xx);
    }
    // update value function derivatives of previous node
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
        d.Q_x *= perm;
        d.V_xx *= perm;
        d.V_xx.applyOnTheLeft(perm.transpose());
        d_pre.Q_y.noalias() += d.Q_x;
        d_pre.V_yy.noalias() += d.V_xx;
    }
    timed_block_end("update_value_function");
}
// only Q_() changed
void generic_solver::riccati_recursion_correction(ns_riccati_data *cur, ns_riccati_data *prev) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    // compute z_u correction
    // nsp.z_u_k.noalias() = d.Q_u.transpose() - nsp.F_u.transpose() * d.Q_y.transpose();
    // nsp.z_u_k.noalias() = d.Q_u.transpose() - d.F_u.dense().transpose() * d.Q_y.transpose();
    // row_vector tmp(d.F_u.cols());
    // tmp.setZero();
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_0_k = d.Q_u.transpose();
        d.F_u.right_times<false>(d.Q_y, nsp.z_0_k);
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K;
        d.F_x.right_times<false>(d.Q_y, d.Q_x);
    } else if (d.rank_status_ == rank_status::constrained) {
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * d.Q_u.transpose() + nsp.Z_y.transpose() * d.Q_y.transpose();
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K - (d.Q_u * nsp.u_y_K + d.Q_y * nsp.y_y_K);
    } else {
        d.Q_x.noalias() += -(d.Q_u * nsp.u_y_K + d.Q_y * nsp.y_y_K);
    }
    // compute Q_x correcton
    // d.Q_x.noalias() = -d.Q_y * nsp.F_0_K + nsp.z_u_k.transpose() * d.d_u.K;
    // d.Q_x.noalias() = -d.Q_y * d.F_x.dense() + nsp.z_u_k.transpose() * d.d_u.K;
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
        d.Q_x *= perm;
        d_pre.V_y_corr = d.Q_x; // save correction to V_y explicitly before accumulating
        d_pre.Q_y += d.Q_x;
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
