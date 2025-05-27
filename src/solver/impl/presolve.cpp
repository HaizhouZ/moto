#include <atri/ocp/approx_storage.hpp>
#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/ns_riccati_solve.hpp>

namespace atri {
namespace ns_riccati {

void pre_solving_steps_0(node *cur) {
    // collect constraint residuals and jacobians
    auto &d = get_data(cur);
    d.dense_->cost_.setZero();
    d.Q_x.setZero();
    d.Q_u.setZero();
    d.Q_y.setZero();
    d.Q_xx.setZero(); 
    d.Q_ux.setZero(); 
    d.Q_uu.setZero(); 
    d.Q_yx.setZero(); 
    d.Q_yy.setZero();
    // update everything
    cur->update_approximation();
}

void pre_solving_steps_1(node *cur) {
    // collect constraint residuals and jacobians
    auto &d = get_data(cur);
    auto &nsp = *d.nsp_;
    auto &_approx = d.dense_->approx_;
    /// @todo sparse F_y inverse
    auto &F = _approx[__dyn].jac_;
    nsp.lu_dyn_.compute(F[__y]);
    // nsp.F_u = F[__u];
    nsp.F_u = nsp.lu_dyn_.solve(F[__u]);
    // nsp.F_0_k = -_approx[__dyn].v_;
    nsp.F_0_k = nsp.lu_dyn_.solve(_approx[__dyn].v_);
    // nsp.F_0_K = -F[__x];
    nsp.F_0_K = nsp.lu_dyn_.solve(F[__x]);
    // nullspace computation
    d.rank_status_ = rank_status::unconstrained;
    if (d.ncstr) {
        if (d.ns) {
            nsp.s_0_p_k.noalias() =
                _approx[__eq_cstr_s].v_ - nsp.s_y * nsp.F_0_k;
            nsp.s_0_p_K.noalias() =
                _approx[__eq_cstr_s].jac_[__x] - nsp.s_y * nsp.F_0_K;
            nsp.s_u.noalias() = -nsp.s_y * nsp.F_u;
            // solve pseudo inverse
            nsp.s_c_stacked.topRows(d.ns) = nsp.s_u;
            nsp.s_c_stacked_0_k.head(d.ns) = nsp.s_0_p_k;
            nsp.s_c_stacked_0_K.topRows(d.ns) = nsp.s_0_p_K;
        }
        if (d.nc) {
            nsp.s_c_stacked.bottomRows(d.nc) = _approx[__eq_cstr_c].jac_[__u];
            nsp.s_c_stacked_0_k.tail(d.nc) = _approx[__eq_cstr_c].v_;
            nsp.s_c_stacked_0_K.bottomRows(d.nc) = _approx[__eq_cstr_c].jac_[__x];
        }
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
void pre_solving_steps_2(node *prev, node *cur) {
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

} // namespace ns_riccati
} // namespace atri