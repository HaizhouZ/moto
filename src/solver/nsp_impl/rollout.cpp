#include <moto/solver/ineq_soft_solve.hpp>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace ns_riccati {
void fwd_linear_rollout(ns_node_data *cur, ns_node_data *next) {
    // get_data(nodes_.front()).prim_step[__x].setZero();
    auto &d = *cur;
    d.prim_step[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_step[__x];
    if (next != nullptr) [[likely]] {
        copy_y_to_x(d.prim_step[__y], next->prim_step[__x], d.prob_, next->prob_);
    }
}
void finalize_newton_step(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.prim_step[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_step[__x];
    ineq_soft_solve::post_rollout(cur);
    // multiplier
    // dynamics multiplier first two terms
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.Q_yx * d.prim_step[__x] - d.Q_yy * d.prim_step[__y];
    // update hard constraint multipliers
    if (d.ncstr > 0 && d.rank_status_ != rank_status::unconstrained) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -nsp.u_0_p_k - nsp.u_0_p_K * d.prim_step[__x] - nsp.U * d.prim_step[__u];
        // fmt::print("u_0_p_k: \n{}\n", nsp.u_0_p_k.transpose());
        // fmt::print("u_0_p_K: \n{}\n", nsp.u_0_p_K.transpose());
        // fmt::print("d.prim_step[__x]: \n{}\n", d.prim_step[__x].transpose());
        // fmt::print("d.prim_step[__u]: \n{}\n", d.prim_step[__u].transpose());
        // for(auto &arg:cur->ocp_->expr_[__y]){
        //     fmt::print("{}({}) ", arg->name_, arg->dim_);
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
} // namespace ns_riccati
} // namespace moto
