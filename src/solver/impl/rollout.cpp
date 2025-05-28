#include <atri/ocp/approx_storage.hpp>
#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/data/rollout_data.hpp>
#include <atri/solver/ns_riccati_solve.hpp>
#include <iostream>

#define a 1

namespace atri {
namespace ns_riccati {
void forward_rollout(node *cur, node *next) {
    // get_data(nodes_.front()).rollout_->prim_[__x].setZero();
    auto &d = get_data(cur);
    auto &rollout_ = *d.rollout_;
    rollout_.prim_[__y].noalias() = d.d_y.k + d.d_y.K * rollout_.prim_[__x];
    if (next != nullptr) [[likely]] {
        get_data(next).rollout_->prim_[__x] = rollout_.prim_[__y];
    }
}
void post_rollout_steps(node *cur) {
    auto &d = get_data(cur);
    auto &rollout_ = *d.rollout_;
    auto &nsp = *d.nsp_;
    rollout_.prim_[__u].noalias() = d.d_u.k + d.d_u.K * rollout_.prim_[__x];
    d.sym_->value_[__x].noalias() += a * rollout_.prim_[__x];
    d.sym_->value_[__u].noalias() += a * rollout_.prim_[__u];
    d.sym_->value_[__y].noalias() += a * rollout_.prim_[__y];
    // multiplier
    // update hard constraint multipliers
    if (d.ncstr > 0) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -nsp.u_0_p_k - nsp.u_0_p_K * rollout_.prim_[__x] - nsp.U * rollout_.prim_[__u];
        // solve for hard constraint multiplers
        d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
    }
    // dynamics multiplier first two terms
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.Q_yx * rollout_.prim_[__x] - d.Q_yy * rollout_.prim_[__y];
    if (d.ns > 0) {
        // append last term in dynamics multipler computation
        d.d_lbd_f.noalias() -= nsp.s_y.transpose() * d.d_lbd_s_c;
        rollout_.dual_[__eq_cstr_s] = d.d_lbd_s_c.head(d.ns);
        d.dense_->dual_[__eq_cstr_s].noalias() += rollout_.dual_[__eq_cstr_s];
    }
    if (d.nc > 0) {
        rollout_.dual_[__eq_cstr_c] = d.d_lbd_s_c.tail(d.nc);
        d.dense_->dual_[__eq_cstr_c].noalias() += rollout_.dual_[__eq_cstr_c];
    }
    rollout_.dual_[__dyn].noalias() = nsp.lu_dyn_.transpose().solve(d.d_lbd_f);
    d.dense_->dual_[__dyn].noalias() += a * rollout_.dual_[__dyn];
}
} // namespace ns_riccati
} // namespace atri
