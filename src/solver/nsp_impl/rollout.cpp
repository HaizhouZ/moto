#include <moto/ocp/dynamics.hpp>
#include <moto/solver/ineq_soft_solve.hpp>
#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/nullspace_data.hpp>

namespace moto {
namespace nullsp_kkt_solve {
void fwd_linear_rollout(riccati_data *cur, riccati_data *next) {
    // get_data(nodes_.front()).prim_step[__x].setZero();
    auto &d = *cur;
    d.prim_step[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_step[__x];
    if (next != nullptr) [[likely]] {
        dynamics::copy_y_to_x(d.prim_step[__y], next->prim_step[__x], d.ocp_, next->ocp_);
    }
}
void finalize_newton_step(riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    d.prim_step[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_step[__x];
    ineq_soft_solve::post_rollout(cur);
    // multiplier
    // update hard constraint multipliers
    if (d.ncstr > 0) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -nsp.u_0_p_k - nsp.u_0_p_K * d.prim_step[__x] - nsp.U * d.prim_step[__u];
        // solve for hard constraint multiplers
        d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
    }
    // dynamics multiplier first two terms
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.Q_yx * d.prim_step[__x] - d.Q_yy * d.prim_step[__y];
    if (d.ns > 0) {
        // append last term in dynamics multipler computation
        d.dual_step[__eq_x] = d.d_lbd_s_c.head(d.ns);
        d.d_lbd_f.noalias() -= nsp.s_y.transpose() * d.dual_step[__eq_x];
    }
    if (d.nc > 0) {
        d.dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
    }
    d.dual_step[__dyn].noalias() = nsp.lu_dyn_.transpose().solve(d.d_lbd_f);
}
} // namespace nullsp_kkt_solve
} // namespace moto
