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
        auto &F = _approx[field::dyn].jac_;
        d.llt_dyn_.compute(F[field::y]);
        d.F_u = F[field::u];
        d.llt_dyn_.solveInPlace(d.F_u);
        d.F_0_k = _approx[field::dyn].v_;
        d.llt_dyn_.solveInPlace(d.F_0_k);
        d.F_0_K = F[field::x];
        d.llt_dyn_.solveInPlace(d.F_0_K);
        // nullspace computation
        d.lu_rank_ = 0;
        if (d.nc + d.ns > 0) {
            d.s_0_p_k.noalias() =
                _approx[field::eq_cstr_s].v_ -
                _approx[field::eq_cstr_s].jac_[field::y] * d.F_0_k;
            d.s_0_p_K.noalias() =
                _approx[field::eq_cstr_s].jac_[field::x] -
                _approx[field::eq_cstr_s].jac_[field::y] * d.F_0_K;
            d.s_u.noalias() = -_approx[field::eq_cstr_s].jac_[field::y] * d.F_u;
            // solve pseudo inverse
            d.s_c_stacked << d.s_u, _approx[field::eq_cstr_c].jac_[field::u];
            d.s_c_stacked_0_k << d.s_0_p_k, _approx[field::eq_cstr_c].v_;
            d.s_c_stacked_0_K << d.s_0_p_K,
                _approx[field::eq_cstr_c].jac_[field::x];
            d.lu_.compute(d.s_c_stacked);
            d.lu_rank_ = d.lu_.rank();
            d.u_y_k.noalias() = d.lu_.solve(d.s_c_stacked_0_k);
            d.u_y_K.noalias() = d.lu_.solve(d.s_c_stacked_0_K);
        }
    }
    // these two cannot merge, because Q_y/yy should first be updated with
    // constr derivatives
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        d.Q_y.noalias() +=
            d.Q_y * d.F_0_K + d.F_0_k.transpose() * d.Q_xy.transpose();
        d.Q_yy.noalias() +=
            d.Q_xy * d.F_0_K + d.F_0_K.transpose() * d.Q_xy.transpose();
    }
}
} // namespace atri