#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
#include <moto/utils/field_conversion.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void fwd_linear_rollout(ns_node_data *cur, ns_node_data *next) {
    // get_data(nodes_.front()).prim_step[__x].setZero();
    auto &d = *cur;
    d.prim_step[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_step[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x(d.prim_step[__y], next->prim_step[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void finalize_dual_newton_step(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.Q_yx * d.prim_step[__x] - d.Q_yy * d.prim_step[__y];
    // update hard constraint multipliers
    if (d.ncstr > 0 && d.rank_status_ != rank_status::unconstrained) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -nsp.u_0_p_k - nsp.u_0_p_K * d.prim_step[__x] - nsp.U * d.prim_step[__u];
        // fmt::print("u_0_p_k: \n{}\n", nsp.u_0_p_k.transpose());
        // fmt::print("u_0_p_K: \n{}\n", nsp.u_0_p_K.transpose());
        // fmt::print("d.prim_step[__x]: \n{}\n", d.prim_step[__x].transpose());
        // fmt::print("d.prim_step[__u]: \n{}\n", d.prim_step[__u].transpose());
        // for(auto &arg:cur->ocp_->exprs(__y)){
        //     fmt::print("{}({}) ", arg->name(), arg->dim());
        // }
        // fmt::print("\nd.Q_y: \n{}\n", d.Q_y.transpose());
        // fmt::print("d.Q_yy: \n{}\n", d.Q_yy.transpose());
        // solve for hard constraint multiplers
        d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
        // fmt::print("d_lbd_s_c_pre_solve: \n{}\n", d.d_lbd_s_c_pre_solve.transpose());
        // fmt::print("d_lbd_s_c: \n{}\n", d.d_lbd_s_c.transpose());
        if (d.ns > 0) {
            // append last term in dynamics multipler computation
            d.dual_step[__eq_x] = d.d_lbd_s_c.head(d.ns);
            d.d_lbd_f.noalias() -= nsp.s_y.transpose() * d.dual_step[__eq_x];
        }
        if (d.nc > 0) {
            d.dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
        }
    }
    d.dual_step[__dyn].noalias() = nsp.lu_dyn_.transpose().solve(d.d_lbd_f);
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
        utils::copy_y_to_x(d.prim_corr[__y], next->prim_corr[__x], cur->dense_->prob_, next->dense_->prob_);
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
    /// correct bar{u}_0 (first order term)
    nsp.u_0_p_k += nsp.z_u_k;
    /// update Q_y with correction
    d.Q_y += *d.Q_y_corr;
    finalize_dual_newton_step(cur);
}
void compute_kkt_residual(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // compute KKT residual
    fmt::println("KKT residuals:");
    matrix res_stat_u = d.Q_uu * d.prim_step[__u] + d.Q_ux * d.prim_step[__x] + d.Q_u.transpose() + d.dense_->approx_[__dyn].jac_[__u].transpose() * d.dual_step[__dyn];
    if (d.nc) {
        res_stat_u += d.dense_->approx_[__eq_xu].jac_[__u].transpose() * d.dual_step[__eq_xu];
    }
    res_stat_u += d.dense_->jac_modification_[__u].transpose();
    fmt::println("res_stat_u: {}", res_stat_u.cwiseAbs().maxCoeff());
    matrix res_stat_y = d.Q_yy * d.prim_step[__y] + d.Q_yx * d.prim_step[__x] + d.Q_y.transpose() + d.dense_->approx_[__dyn].jac_[__y].transpose() * d.dual_step[__dyn];
    if (d.ns) {
        res_stat_y += d.dense_->approx_[__eq_x].jac_[__y].transpose() * d.dual_step[__eq_x];
    }
    fmt::println("res_stat_y: {}", res_stat_y.cwiseAbs().maxCoeff());
    matrix res_dyn = d.dense_->approx_[__dyn].v_ + d.dense_->approx_[__dyn].jac_[__x] * d.prim_step[__x] +
                     d.dense_->approx_[__dyn].jac_[__u] * d.prim_step[__u] + d.dense_->approx_[__dyn].jac_[__y] * d.prim_step[__y];
    fmt::println("res_dyn: {}", res_dyn.cwiseAbs().maxCoeff());
    if (d.ns) {
        matrix res_eq_x = d.dense_->approx_[__eq_x].v_ + d.dense_->approx_[__eq_x].jac_[__y] * d.prim_step[__y];
        fmt::println("res_eq_x: {}", res_eq_x.cwiseAbs().maxCoeff());
    }
    if (d.nc) {
        matrix res_eq_xu = d.dense_->approx_[__eq_xu].v_ + d.dense_->approx_[__eq_xu].jac_[__x] * d.prim_step[__x] +
                           d.dense_->approx_[__eq_xu].jac_[__u] * d.prim_step[__u];
        fmt::println("res_eq_xu: {}", res_eq_xu.cwiseAbs().maxCoeff());
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
