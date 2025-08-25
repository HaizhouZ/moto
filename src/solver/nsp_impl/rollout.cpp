#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
#include <moto/utils/field_conversion.hpp>
#include <Eigen/Eigenvalues>

namespace moto {
namespace solver {
namespace ns_riccati {
void fwd_linear_rollout(ns_node_data *cur, ns_node_data *next) {
    // get_data(nodes_.front()).prim_step[__x].setZero();
    auto &d = *cur;
    d.prim_step[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_step[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x_tangent(d.prim_step[__y], next->prim_step[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void finalize_dual_newton_step(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.Q_yx * d.prim_step[__x] - d.Q_yy * d.prim_step[__y];
    // update hard constraint multipliers
    if (d.ncstr > 0 && d.rank_status_ != rank_status::unconstrained) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -d.Q_u.transpose() - d.Q_ux * d.prim_step[__x] - d.Q_uu * d.prim_step[__u];
        d.F_u.T_times<false>(d.d_lbd_f, d.d_lbd_s_c_pre_solve);
        // solve for hard constraint multiplers
        // fmt::print("Q_y: \n{}\n", d.Q_y);
        // fmt::print("Q_zz: \n{}\n", nsp.Q_zz);
        // fmt::print("Q_zz eigenvalues: {}\n", nsp.Q_zz.eigenvalues().transpose());
        // fmt::print("\n");
        // fmt::print("Q_u: \n{}\n", d.Q_u);
        // fmt::print("Q_yy: \n{}\n", d.Q_yy);
        // // fmt::print("Z_u: \n{}\n", nsp.Z_u);
        // // fmt::print("Z_y: \n{}\n", nsp.Z_y);
        // fmt::print("Z_k: \n{}\n", nsp.z_K);
        // // fmt::print("y_y_K: \n{}\n", nsp.y_y_K);
        // fmt::print("y_0_p_k: \n{}\n", nsp.y_0_p_k.transpose());
        // fmt::print("y_0_p_K: \n{}\n", nsp.y_0_p_K);
        // fmt::print("d_lbd_f: \n{}\n", d.d_lbd_f.transpose());
        // fmt::print("d_lbd_s_c_pre_solve: \n{}\n", d.d_lbd_s_c_pre_solve.transpose());
        d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
        size_t cur_idx = 0;
        if (d.ns > 0) {
            // append last term in dynamics multipler computation
            d.dual_step[__eq_x] = d.d_lbd_s_c.head(d.ns);
            cur_idx += d.ns;
            // d.d_lbd_f.noalias() -= nsp.s_y.transpose() * d.dual_step[__eq_x];
            d.s_y.T_times<false>(d.dual_step[__eq_x], d.d_lbd_f);
        }
        if (d.nc > 0) {
            d.dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
        }
    }
    // d.dual_step[__dyn].noalias() =  nsp.lu_dyn_.transpose().solve(d.d_lbd_f);
    cur->apply_jac_y_inverse_transpose(d.d_lbd_f, d.dual_step[__dyn]);
}
void finalize_newton_step(ns_node_data *cur, bool finalize_dual) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.prim_step[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_step[__x];
    // multiplier
    // dynamics multiplier first two terms
    if (finalize_dual)
        finalize_dual_newton_step(cur);
}
void fwd_linear_rollout_correction(ns_node_data *cur, ns_node_data *next) {
    // get_data(nodes_.front()).prim_step[__x].setZero();
    auto &d = *cur;
    d.prim_corr[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_corr[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x_tangent(d.prim_corr[__y], next->prim_corr[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void finalize_newton_step_correction(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.prim_corr[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_corr[__x];
    // correction for the primal step
    for (auto f : primal_fields) {
        d.prim_step[f] += d.prim_corr[f];
    }
    /// update Q_y with correction
    d.Q_u += d.dense_->jac_modification_[__u];
    d.Q_y += d.dense_->jac_modification_[__y];
}
void compute_kkt_residual(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // compute KKT residual
    // fmt::println("KKT residuals:");
    auto dense = d.dense_;
    // fmt::println("Q_y: {}", d.Q_y);
    auto &f_u = dense->approx_[__dyn].jac_[__u];
    dense->res_stat_[__u].noalias() = d.Q_uu_bak * d.prim_step[__u] + d.Q_u_bak.transpose(); // d.d_lbd_f
    f_u.right_T_times(d.dual_step[__dyn], dense->res_stat_[__u]);
    // dense->res_stat_[__u].noalias() += d.dual_step[__dyn].transpose() * f_u.dense();
    if (d.nc) {
        d.dense_->approx_[__eq_xu].jac_[__u].right_T_times(d.dual_step[__eq_xu], dense->res_stat_[__u]);
    }
    if (d.dual_step[__ineq_xu].size() > 0) {
        d.dense_->approx_[__ineq_xu].jac_[__u].right_T_times(d.dual_step[__ineq_xu], dense->res_stat_[__u]);
    }
    dense->res_stat_[__y].noalias() = d.Q_yy_bak * d.prim_step[__y] + d.Q_y_bak.transpose();
    auto &f_y = dense->approx_[__dyn].jac_[__y];
    f_y.right_T_times(d.dual_step[__dyn], dense->res_stat_[__y]);
    // dense->res_stat_[__y].noalias() += d.dual_step[__dyn].transpose() * f_y.dense();
    if (d.ns) {
        d.dense_->approx_[__eq_x].jac_[__y].right_T_times(d.dual_step[__eq_x], dense->res_stat_[__y]);
    }
    if (d.dual_step[__ineq_x].size() > 0) {
        d.dense_->approx_[__ineq_x].jac_[__y].right_T_times(d.dual_step[__ineq_x], dense->res_stat_[__y]);
    }
    auto &f_x = dense->approx_[__dyn].jac_[__x];
    dense->res_stat_[__x].noalias() = d.Q_xx_bak * d.prim_step[__x] + d.Q_x_bak.transpose();
    f_x.right_T_times(d.dual_step[__dyn], dense->res_stat_[__x]);
    // dense->res_stat_[__x].noalias() += d.dual_step[__dyn].transpose() * f_x.dense();
    if (d.nc) {
        d.dense_->approx_[__eq_xu].jac_[__x].right_T_times(d.dual_step[__eq_xu], dense->res_stat_[__x]);
    }
    if (d.dual_step[__ineq_xu].size() > 0) {
        d.dense_->approx_[__ineq_xu].jac_[__x].right_T_times(d.dual_step[__ineq_xu], dense->res_stat_[__x]);
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
