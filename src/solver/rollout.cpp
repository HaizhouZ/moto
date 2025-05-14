#include <atri/ocp/core/approx_data.hpp>
#include <atri/solver/data/rollout_data.hpp>
#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {
namespace ns_riccati_solver {
void forward_rollout(shooting_node *cur, shooting_node *next) {
    // get_data(nodes_.front()).rollout_->prim_[__x].setZero();
    auto &d = get_data(cur);
    auto &rollout_ = *d.rollout_;
    rollout_.prim_[__y].noalias() = d.d_y.k + d.d_y.K * rollout_.prim_[__x];
    if (next != nullptr) [[likely]] {
        get_data(next).rollout_->prim_[__x] = rollout_.prim_[__y];
    }
}
void post_rollout_steps(shooting_node *cur) {
    auto &d = get_data(cur);
    auto &rollout_ = *d.rollout_;
    rollout_.prim_[__u].noalias() = d.d_u.k + d.d_u.K * rollout_.prim_[__x];
    d.sym_->value_[__x] += rollout_.prim_[__x];
    d.sym_->value_[__u] += rollout_.prim_[__u];
    d.sym_->value_[__y] += rollout_.prim_[__y];
    rollout_.dual_[__dyn].noalias() =
        d.d_lbd_f.k + d.d_lbd_f.K * rollout_.prim_[__x];
    d.raw_->dual_[__dyn] += rollout_.dual_[__dyn];
    if (d.ns > 0) {
        rollout_.dual_[__eq_cstr_s].noalias() =
            d.d_lbd_s_c.k.head(d.ns) +
            d.d_lbd_s_c.K.topRows(d.ns) * rollout_.prim_[__x];

        d.raw_->dual_[__eq_cstr_s] += rollout_.dual_[__eq_cstr_s];
    }
    if (d.nc > 0) {
        rollout_.dual_[__eq_cstr_c].noalias() =
            d.d_lbd_s_c.k.tail(d.nc) +
            d.d_lbd_s_c.K.bottomRows(d.nc) * rollout_.prim_[__x];
        d.raw_->dual_[__eq_cstr_c] += rollout_.dual_[__eq_cstr_c];
    }
    pre_solving_steps_1(cur);
}
} // namespace ns_riccati_solver
} // namespace atri
