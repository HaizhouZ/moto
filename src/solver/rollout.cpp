#include <atri/solver/ns_riccati_solver.hpp>
namespace atri {
void nullspace_riccati_solver::forward_rollout() {
    get_data(nodes_.front()).rollout_.prim_[__x].setZero();
    for (int i = 0; i < nodes_.size(); i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        if (i != 0) {
            d.rollout_.prim_[__x] = get_data(nodes_[i - 1]).rollout_.prim_[__y];
        }
        d.rollout_.prim_[__y].noalias() =
            d.d_y.k + d.d_y.K * d.rollout_.prim_[__x];
    }
}
void nullspace_riccati_solver::post_rollout_steps() {
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        d.rollout_.prim_[__u].noalias() =
            d.d_u.k + d.d_u.K * d.rollout_.prim_[__x];
        d.rollout_.dual_[__dyn].noalias() =
            d.d_lbd_f.k + d.d_lbd_f.K * d.rollout_.prim_[__x];
        if (d.ns > 0)
            d.rollout_.dual_[__eq_cstr_s].noalias() =
                d.d_lbd_s_c.k.head(d.ns) +
                d.d_lbd_s_c.K.topRows(d.ns) * d.rollout_.prim_[__x];
        if (d.nc > 0)
            d.rollout_.dual_[__eq_cstr_c].noalias() =
                d.d_lbd_s_c.k.tail(d.nc) +
                d.d_lbd_s_c.K.bottomRows(d.nc) * d.rollout_.prim_[__x];
    }
}
} // namespace atri
