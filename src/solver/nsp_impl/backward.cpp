#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
#include <moto/utils/field_conversion.hpp>

// #define ENABLE_TIMED_BLOCK
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
    if (d.Q_yy.hasNaN()) {
        fmt::print("Q_yy: \n {}\n", d.Q_yy);
        print_debug(cur);
        throw std::runtime_error("Q_yy has NaN");
    }
    // matrix::AlignedMapType F_u(d.F_u.dense_panels_[0].mat().data(), d.F_u.rows(), d.F_u.cols());
    // nsp.U.noalias() = d.Q_uu + F_u.transpose() * d.Q_yy * F_u;
    timed_block_start("compute_U");
    nsp.U.noalias() = d.Q_uu;
    d.F_u.inner_product(d.Q_yy, nsp.U);
    // compute bar{u}_0
    nsp.Q_yy_F_0.noalias() = d.Q_yy * d.F_0 - d.Q_y.transpose();
    // nsp.u_0_p_k.noalias() = d.Q_u.transpose() - nsp.F_u.transpose() * (d.Q_y.transpose() - d.Q_yy * nsp.F_0_k);
    // nsp.u_0_p_k.noalias() = d.Q_u.transpose() + d.F_u.dense().transpose() * (d.Q_yy * d.F_0 - d.Q_y.transpose());
    d.F_u.T_times(nsp.Q_yy_F_0, nsp.u_0_p_k);
    // nsp.Q_yy_F_x.noalias() = d.Q_yy * F_x;
    d.F_x.right_times(d.Q_yy, nsp.Q_yy_F_x);
    // nsp.u_0_p_K.noalias() = d.Q_ux + d.F_u.dense().transpose() * (nsp.Q_yy_F_x - d.Q_yx);
    d.F_u.T_times(nsp.Q_yy_F_x, nsp.u_0_p_K);
    timed_block_end("compute_U");
    // fmt::print("u_0_p_k: \n{}\n", nsp.u_0_p_k.transpose());
    // fmt::print("u_0_p_K: \n{}\n", nsp.u_0_p_K.transpose());
    // fmt::print("u_y_K: \n{}\n", nsp.u_y_K.transpose());
    // compute z_u
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_u_k.noalias() = nsp.u_0_p_k;
        nsp.z_u_K.noalias() = nsp.u_0_p_K;
        d.d_u.K.noalias() = -nsp.z_u_K;
        nsp.llt_ns_.compute(nsp.U);
        if (nsp.llt_ns_.info() != Eigen::Success) {
            fmt::print("Q_uu: \n{}\n", d.Q_uu);
            fmt::print("Q_yy: \n{}\n", d.Q_yy);
            fmt::print("U: \n{}\n", nsp.U);
            kkt_diagnosis(cur);
            print_debug(cur);
            throw std::runtime_error("U is not positive definite");
        } else
            nsp.llt_ns_.solveInPlace(d.d_u.K);
    } else {
        // generic_constr rank > 0
        timed_block_start("compute_z_u");
        nsp.z_u_k.noalias() = nsp.u_0_p_k - nsp.U * nsp.u_y_k;
        nsp.z_u_K.noalias() = nsp.u_0_p_K - nsp.U * nsp.u_y_K;
        if (d.rank_status_ == rank_status::fully_constrained) {
            // fully constrained
        } else {
            // solve nullspace system
            nsp.U_z.noalias() = nsp.Z.transpose() * nsp.U * nsp.Z;
            nsp.u_z_K.noalias() = -nsp.Z.transpose() * nsp.z_u_K;
            timed_block_end("compute_z_u");
            nsp.llt_ns_.compute(nsp.U_z);
            nsp.llt_ns_.solveInPlace(nsp.u_z_K);
            d.d_u.K.noalias() = nsp.Z * nsp.u_z_K - nsp.u_y_K;
        }
    }
    timed_block_start("update_value_function");
    // d.Q_x.noalias() += -d.Q_y * d.F_x.dense() + d.F_0.transpose() * nsp.Q_yy_F_x +
    //                    nsp.z_u_k.transpose() * d.d_u.K;
    d.F_x.right_times<false>(d.Q_y, d.Q_x);
    d.Q_x.noalias() += d.F_0.transpose() * nsp.Q_yy_F_x + nsp.z_u_k.transpose() * d.d_u.K;
    // d.Q_xx.noalias() += d.F_x.dense().transpose() * nsp.Q_yy_F_x + nsp.z_u_K.transpose() * d.d_u.K;
    d.F_x.T_times(nsp.Q_yy_F_x, d.Q_xx);
    d.Q_xx.noalias() += nsp.z_u_K.transpose() * d.d_u.K;
    if (d.rank_status_ != rank_status::unconstrained) {
        d.Q_x.noalias() -= nsp.u_y_k.transpose() * nsp.u_0_p_K;
        d.Q_xx.noalias() -= nsp.u_y_K.transpose() * nsp.u_0_p_K;
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
    nsp.z_u_k = d.Q_u.transpose(); // + tmp.transpose();
    d.F_u.right_times<false>(d.Q_y, nsp.z_u_k);
    // compute Q_x correcton
    // d.Q_x.noalias() = -d.Q_y * nsp.F_0_K + nsp.z_u_k.transpose() * d.d_u.K;
    // d.Q_x.noalias() = -d.Q_y * d.F_x.dense() + nsp.z_u_k.transpose() * d.d_u.K;
    d.Q_x.noalias() = nsp.z_u_k.transpose() * d.d_u.K;
    d.F_x.right_times<false>(d.Q_y, d.Q_x);
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
