#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
#include <moto/utils/field_conversion.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
extern void kkt_diagnosis(ns_node_data *cur);
extern void print_debug(ns_node_data *cur);
void riccati_recursion(ns_node_data *cur, ns_node_data *prev) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // check positiveness
    // bool qyy_invertible = isPositiveDefinite(d.Q_yy);
    timed_block_start("Q_yy symmetrize");
    d.Q_yy.array() /= 2;
    /// @todo: temporary
    d.Q_yy = d.Q_yy + d.Q_yy.transpose().eval();
    timed_block_end("Q_yy symmetrize");
    // d.Q_yy.diagonal().array() += 1e-6; // ensure positive definiteness
    if (d.Q_yy.hasNaN() || !d.Q_yy.allFinite()) {
        fmt::print("Q_yy: \n {}\n", d.Q_yy);
        print_debug(cur);
        throw std::runtime_error("Q_yy has NaN or inf");
    }
    // if (d.Q_yy.llt().info() != Eigen::Success) {
    //     fmt::print("Q_yy: \n {}\n", d.Q_yy);
    //     // print eigenvalues
    //     auto eigs = d.Q_yy.eigenvalues();
    //     fmt::print("Q_yy eigenvalues: {}\n", eigs.transpose());
    //     print_debug(cur);
    //     throw std::runtime_error("Q_yy is not positive definite");
    // }
    timed_block_start("compute_U");
    nsp.y_0_p_k.noalias() = d.Q_y.transpose() - d.Q_yy * nsp.y_y_k;
    if (d.rank_status_ == rank_status::unconstrained) {
        d.F_x.right_times<false>(d.Q_yy, nsp.y_0_p_K);
        d.F_u.inner_product(d.Q_yy, nsp.Q_zz);
        d.F_u.T_times<false>(nsp.y_0_p_k, nsp.z_0_k);
        d.F_u.T_times<false>(nsp.y_0_p_K, nsp.z_0_K);
    } else if (d.rank_status_ == rank_status::constrained) {
        nsp.y_0_p_K.noalias() = d.Q_yx - d.Q_yy * nsp.y_y_K;
        nsp.Q_zz.noalias() += nsp.Z_y.transpose() * d.Q_yy * nsp.Z_y; /// todo unconstrained
        // compute bar{y}_0
        nsp.z_0_k.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_k;
        nsp.z_0_K.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_K;
    }
    timed_block_end("compute_U");
    // auto min_Q_uu = d.Q_uu.eigenvalues().real().minCoeff();
    // auto min_Q_yy = d.Q_yy.eigenvalues().real().minCoeff();
    // auto min_Q_zz = nsp.Q_zz.eigenvalues().real().minCoeff();
    // if (min_Q_uu <= 0 || min_Q_yy <= 0 || min_Q_zz <= 0) {
    //     fmt::print("Q_uu min eigenvalue: {}\n", min_Q_uu);
    //     fmt::print("Q_yy min eigenvalue: {}\n", min_Q_yy);
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
            fmt::print("Q_uu: \n{}\n", d.Q_uu);
            fmt::print("Q_yy: \n{}\n", d.Q_yy);
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
        } else {
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
        d.Q_xx.noalias() += -nsp.y_0_p_K.transpose() * nsp.y_y_K;
    } else if (d.rank_status_ == rank_status::constrained) {
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K - nsp.y_0_p_k.transpose() * nsp.y_y_K;
        d.Q_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K - nsp.y_0_p_K.transpose() * nsp.y_y_K;
    } else {
        d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K;
        d.F_x.right_T_times<false>(nsp.y_0_p_k, d.Q_x);
        d.Q_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K;
        d.F_x.right_T_times<false>(nsp.y_0_p_K, d.Q_xx);
    }
    // update value function derivatives of previous node
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
        d.Q_x *= perm;
        d.Q_xx *= perm;
        d.Q_xx.applyOnTheLeft(perm.transpose());
        d_pre.Q_y.noalias() += d.Q_x;
        d_pre.Q_yy.noalias() += d.Q_xx;
    }
    timed_block_end("update_value_function");
}
// only Q_() changed
void riccati_recursion_correction(ns_node_data *cur, ns_node_data *prev) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // compute z_u correction
    // nsp.z_u_k.noalias() = d.Q_u.transpose() - nsp.F_u.transpose() * d.Q_y.transpose();
    // nsp.z_u_k.noalias() = d.Q_u.transpose() - d.F_u.dense().transpose() * d.Q_y.transpose();
    // row_vector tmp(d.F_u.cols());
    // tmp.setZero();
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_0_k = d.Q_u.transpose();
        d.F_u.right_times<false>(d.Q_y, nsp.z_0_k);
        d.Q_x.noalias() = nsp.z_0_k.transpose() * nsp.z_K;
        d.F_x.right_times<false>(d.Q_y, d.Q_x);
    } else if (d.rank_status_ == rank_status::constrained) {
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * d.Q_u.transpose() + nsp.Z_y.transpose() * d.Q_y.transpose();
        d.Q_x.noalias() = nsp.z_0_k.transpose() * nsp.z_K - (d.Q_u * nsp.u_y_K + d.Q_y * nsp.y_y_K);
    } else {
        d.Q_x.noalias() = -(d.Q_u * nsp.u_y_K + d.Q_y * nsp.y_y_K);
    }
    // compute Q_x correcton
    // d.Q_x.noalias() = -d.Q_y * nsp.F_0_K + nsp.z_u_k.transpose() * d.d_u.K;
    // d.Q_x.noalias() = -d.Q_y * d.F_x.dense() + nsp.z_u_k.transpose() * d.d_u.K;
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
        d.Q_x *= perm;
        d_pre.Q_y += d.Q_x;
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
