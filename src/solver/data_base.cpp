#include <moto/solver/data_base.hpp>

namespace moto {
namespace solver {
data_base::data_base(sym_data* s, dense_approx_data *dense)
    : nx(dense->jac_[__x].size()),
      nu(dense->jac_[__u].size()),
      ny(dense->jac_[__y].size()),
      sym_(s), dense_(dense),
      Q_x(dense->jac_[__x]), Q_u(dense->jac_[__u]),
      Q_y(dense->jac_[__y]), Q_xx(dense->hessian_[__x][__x]),
      Q_ux(dense->hessian_[__u][__x]), Q_uu(dense->hessian_[__u][__u]),
      Q_yx(dense->hessian_[__y][__x]), Q_yy(dense->hessian_[__y][__y]) {
    prim_step[__x].resize(nx);
    prim_step[__x].setZero();
    prim_step[__u].resize(nu);
    prim_step[__y].resize(ny);
    prim_corr[__x].resize(nx);
    prim_corr[__x].setZero();
    prim_corr[__u].resize(nu);
    prim_corr[__y].resize(ny);
    Q_y_corr = nullptr;
}
void data_base::merge_jacobian_modification() {
    for (const auto &field : primal_fields) {
        dense_->jac_[field] += dense_->jac_modification_[field];
    }
}

void data_base::swap_jacobian_modification() {
    for (const auto &field : primal_fields) {
        dense_->jac_[field].swap(dense_->jac_modification_[field]);
    }
}

} // namespace solver
} // namespace moto
