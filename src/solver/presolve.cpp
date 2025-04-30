#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_riccati_solver::pre_solving_steps() {
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        // collect constraint residuals and jacobians
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        for (auto m : {d.Q_x, d.Q_u, d.Q_y}) {
            m.setZero();
        }
        for (auto m : {d.Q_xx, d.Q_xu, d.Q_uu, d.Q_xy, d.Q_yy}) {
            m.setZero();
        }
        auto &_approx = d.raw_data_.approx_;
        // update everything
        cur->update_approximation();
        /// @todo sparse F_y inverse
        auto &F = _approx[__dyn].jac_;
        d.llt_dyn_.compute(F[__y]);
        d.F_u = F[__u];
        d.llt_dyn_.solveInPlace(d.F_u);
        d.F_0_k = -_approx[__dyn].v_;
        d.llt_dyn_.solveInPlace(d.F_0_k);
        d.F_0_K = -F[__x];
        d.llt_dyn_.solveInPlace(d.F_0_K);
        // nullspace computation
        d.rank_status_ = rank_status::unconstrained;
        if (d.nc + d.ns > 0) {
            d.s_0_p_k.noalias() =
                -_approx[__eq_cstr_s].v_ - d.s_y * d.F_0_k;
            d.s_0_p_K.noalias() =
                -_approx[__eq_cstr_s].jac_[__x] - d.s_y * d.F_0_K;
            d.s_u.noalias() = -d.s_y * d.F_u;
            // solve pseudo inverse
            d.s_c_stacked << d.s_u, _approx[__eq_cstr_c].jac_[__u];
            d.s_c_stacked_0_k << d.s_0_p_k, -_approx[__eq_cstr_c].v_;
            d.s_c_stacked_0_K << d.s_0_p_K,
                -_approx[__eq_cstr_c].jac_[__x];
            d.lu_eq_.compute(d.s_c_stacked);
            size_t rank = d.lu_eq_.rank();
            if (rank == 0)
                d.rank_status_ = rank_status::unconstrained;
            else if (rank == d.ncstr) {
                d.rank_status_ = rank_status::fully_constrained;
            } else {
                d.Z = d.lu_eq_.kernel();
                d.rank_status_ = rank_status::constrained;
            }
            d.u_y_k.noalias() = d.lu_eq_.solve(d.s_c_stacked_0_k);
            d.u_y_K.noalias() = d.lu_eq_.solve(d.s_c_stacked_0_K);
        }
    }
    // these two cannot merge, because Q_y/yy should first be updated with
    // constr derivatives
#pragma omp parallel
    for (int i = nodes_.size(); i > 0; i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        auto pre = nodes_[i - 1];
        auto &d_pre = get_data(cur);
        // add P part of V_x/V_xx to Q_y/Q_yy of previous node
        d_pre.Q_y.noalias() += d.F_0_k.transpose() * -d.Q_xy.transpose();
        // +d.Q_y * d.F_0_K is done in backward pass
        // because Q_y has V_y in it
        d_pre.Q_yy.noalias() +=
            -d.Q_xy * d.F_0_K + d.F_0_K.transpose() * -d.Q_xy.transpose();
    }
    /// @todo set terminal Q_y, Q_yy
}
} // namespace atri