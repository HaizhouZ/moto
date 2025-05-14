#include <atri/ocp/core/approx_data.hpp>
#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {
namespace ns_riccati_solver {

void pre_solving_steps_0(shooting_node *cur) {
    // collect constraint residuals and jacobians
    auto &d = get_data(cur);
    auto &nsp = *d.nsp_;
    d.raw_->cost_.setZero();
    for (auto m : {d.Q_x, d.Q_u, d.Q_y}) {
        m.setZero();
    }
    for (auto m : {d.Q_xx, d.Q_ux, d.Q_uu, d.Q_yx, d.Q_yy}) {
        m.setZero();
    }
    auto &_approx = d.raw_->approx_;
    // update everything
    cur->update_approximation();
}

void pre_solving_steps_1(shooting_node *cur) {
    // collect constraint residuals and jacobians
    auto &d = get_data(cur);
    auto &nsp = *d.nsp_;
    auto &_approx = d.raw_->approx_;
    /// @todo sparse F_y inverse
    auto &F = _approx[__dyn].jac_;
    nsp.llt_dyn_.compute(F[__y]);
    nsp.F_u = F[__u];
    nsp.llt_dyn_.solveInPlace(nsp.F_u);
    nsp.F_0_k = -_approx[__dyn].v_;
    nsp.llt_dyn_.solveInPlace(nsp.F_0_k);
    nsp.F_0_K = -F[__x];
    nsp.llt_dyn_.solveInPlace(nsp.F_0_K);
    // nullspace computation
    d.rank_status_ = rank_status::unconstrained;
    if (d.nc + d.ns > 0) {
        nsp.s_0_p_k.noalias() =
            -_approx[__eq_cstr_s].v_ - nsp.s_y * nsp.F_0_k;
        nsp.s_0_p_K.noalias() =
            -_approx[__eq_cstr_s].jac_[__x] - nsp.s_y * nsp.F_0_K;
        nsp.s_u.noalias() = -nsp.s_y * nsp.F_u;
        // solve pseudo inverse
        nsp.s_c_stacked << nsp.s_u, _approx[__eq_cstr_c].jac_[__u];
        nsp.s_c_stacked_0_k << nsp.s_0_p_k, -_approx[__eq_cstr_c].v_;
        nsp.s_c_stacked_0_K << nsp.s_0_p_K, -_approx[__eq_cstr_c].jac_[__x];
        nsp.lu_eq_.compute(nsp.s_c_stacked);
        size_t rank = nsp.lu_eq_.rank();
        if (rank == 0)
            d.rank_status_ = rank_status::unconstrained;
        else if (rank == d.ncstr) {
            d.rank_status_ = rank_status::fully_constrained;
        } else {
            nsp.Z = nsp.lu_eq_.kernel();
            d.rank_status_ = rank_status::constrained;
        }
        nsp.u_y_k.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_k);
        nsp.u_y_K.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_K);
    }
}

// these two cannot merge, because Q_y/yy should first be updated with
// constr derivatives
void pre_solving_steps_2(shooting_node *prev, shooting_node *cur) {
    auto &d = get_data(cur);
    auto &d_pre = get_data(prev);
    auto &nsp = *d.nsp_;
    // add P part of V_x/V_xx to Q_y/Q_yy of previous node
    d_pre.Q_y.noalias() += d.Q_x - nsp.F_0_k.transpose() * d.Q_yx;
    // +d.Q_y * d.F_0_K is done in backward pass
    // because Q_y has V_y in it
    d_pre.Q_yy.noalias() +=
        d.Q_xx - (d.Q_yx.transpose() * nsp.F_0_K + nsp.F_0_K.transpose() * d.Q_yx);
}
/// @todo set terminal Q_y, Q_yy

} // namespace ns_riccati_solver
} // namespace atri