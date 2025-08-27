#include <moto/solver/data_base.hpp>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace solver {
data_base::data_base(sym_data *s, merit_data *dense)
    : nx(dense->jac_[__x].size()),
      nu(dense->jac_[__u].size()),
      ny(dense->jac_[__y].size()),
      sym_(s), dense_(dense),
      Q_x(dense->jac_[__x]), Q_u(dense->jac_[__u]),
      Q_y(dense->jac_[__y]), Q_xx(dense->hessian_[__x][__x]),
      Q_ux(dense->hessian_[__u][__x]), Q_uu(dense->hessian_[__u][__u]),
      Q_yx(dense->hessian_[__y][__x]), Q_yy(dense->hessian_[__y][__y]),
      Q_xx_mod(dense->hessian_modification_[__x][__x]),
      Q_ux_mod(dense->hessian_modification_[__u][__x]),
      Q_uu_mod(dense->hessian_modification_[__u][__u]),
      Q_yx_mod(dense->hessian_modification_[__y][__x]),
      Q_yy_mod(dense->hessian_modification_[__y][__y]) {
    V_xx.resize(nx, nx);
    V_xx.setZero();
    V_yy.resize(ny, ny);
    V_yy.setZero();
    for (auto f : primal_fields) {
        prim_step[f].resize(dense->jac_[f].size());
        prim_step[f].setZero();
        prim_corr[f].resize(dense->jac_[f].size());
        prim_corr[f].setZero();
    }
    // set rollout data for constraints
    for (auto f : constr_fields) {
        dual_step[f].resize(dense->approx_[f].v_.size());
        dual_step[f].setZero();
    }
}
void data_base::merge_jacobian_modification() {
    timed_block_start("backup_jacobian");
    Q_u_bak = Q_u; // backup
    Q_x_bak = Q_x;
    Q_y_bak = Q_y;
    timed_block_end("backup_jacobian");
    timed_block_start("merge_jacobian_modification");
    for (const auto &field : primal_fields) {
        dense_->jac_[field] += dense_->jac_modification_[field];
    }
    timed_block_end("merge_jacobian_modification");
    // timed_block_start("backup_hessian");
    // Q_uu_bak = Q_uu; // backup
    // Q_yy_bak = Q_yy;
    // Q_xx_bak = Q_xx;
    // timed_block_end("backup_hessian");
    // timed_block_start("merge_hessian_modification");
    // for (size_t i = 0; i < field::num_prim; i++) {
    //     for (size_t j = i; j < field::num_prim; j++) {
    //         dense_->hessian_[j][i] += dense_->hessian_modification_[j][i];
    //     }
    // }
    // timed_block_end("merge_hessian_modification");
}

void data_base::swap_jacobian_modification() {
    for (const auto &field : primal_fields) {
        dense_->jac_[field].swap(dense_->jac_modification_[field]);
    }
}

} // namespace solver
} // namespace moto
