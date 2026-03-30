#include <moto/ocp/impl/sym_data.hpp>
#include <moto/solver/data_base.hpp>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace solver {
data_base::data_base(sym_data *s, lag_data *dense)
    : nx(dense->lag_jac_[__x].size()),
      nu(dense->lag_jac_[__u].size()),
      ny(dense->lag_jac_[__y].size()),
      sym_(s), dense_(dense),
      Q_x(dense->lag_jac_[__x]), Q_u(dense->lag_jac_[__u]),
      Q_y(dense->lag_jac_[__y]), Q_xx(dense->lag_hess_[__x][__x]),
      Q_ux(dense->lag_hess_[__u][__x]), Q_uu(dense->lag_hess_[__u][__u]),
      Q_yx(dense->lag_hess_[__y][__x]), Q_yy(dense->lag_hess_[__y][__y]),
      Q_xx_mod(dense->hessian_modification_[__x][__x]),
      Q_ux_mod(dense->hessian_modification_[__u][__x]),
      Q_uu_mod(dense->hessian_modification_[__u][__u]),
      Q_yx_mod(dense->hessian_modification_[__y][__x]),
      Q_yy_mod(dense->hessian_modification_[__y][__y]) {
    for (auto f : primal_fields) {
        base_lag_grad_backup[f].resize(dense->lag_jac_[f].size());
        base_lag_grad_backup[f].setZero();
        kkt_stat_err_[f].resize(dense->lag_jac_[f].size());
        kkt_stat_err_[f].setZero();
    }
    V_xx.resize(nx, nx);
    V_xx.setZero();
    V_yy.resize(ny, ny);
    V_yy.setZero();
    for (auto f : primal_fields) {
        trial_prim_step[f].resize(dense->lag_jac_[f].size());
        trial_prim_step[f].setZero();
        prim_corr[f].resize(dense->lag_jac_[f].size());
        prim_corr[f].setZero();
        trial_prim_state_bak[f].resize(dense->lag_jac_[f].size());
        trial_prim_state_bak[f].setZero();
    }
    // set rollout data for constraints
    for (auto f : constr_fields) {
        trial_dual_step[f].resize(dense->approx_[f].v_.size());
        trial_dual_step[f].setZero();
        trial_dual_state_bak[f].resize(dense->approx_[f].v_.size());
        trial_dual_state_bak[f].setZero();
    }
}
void data_base::activate_lag_jac_corr() {
    timed_block_start("backup_jacobian");
    base_lag_grad_backup[__u] = Q_u;
    base_lag_grad_backup[__x] = Q_x;
    base_lag_grad_backup[__y] = Q_y;
    timed_block_end("backup_jacobian");

    timed_block_start("activate_lag_jac_corr");
    for (const auto &field : primal_fields) {
        dense_->lag_jac_[field] += dense_->lag_jac_corr_[field];
    }
    timed_block_end("activate_lag_jac_corr");
}

void data_base::swap_active_and_lag_jac_corr() {
    for (const auto &field : primal_fields) {
        dense_->lag_jac_[field].swap(dense_->lag_jac_corr_[field]);
    }
}

void data_base::backup_trial_state() {
    for (auto field : primal_fields) {
        trial_prim_state_bak[field] = sym_->value_[field];
    }
    for (auto field : constr_fields) {
        trial_dual_state_bak[field] = dense_->dual_[field];
    }
}

void data_base::restore_trial_state() {
    for (auto field : primal_fields) {
        sym_->value_[field] = trial_prim_state_bak[field];
    }
    for (auto field : constr_fields) {
        dense_->dual_[field] = trial_dual_state_bak[field];
    }
}

} // namespace solver
} // namespace moto
