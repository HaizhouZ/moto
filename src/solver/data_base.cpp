#include <moto/solver/data_base.hpp>

namespace moto {
namespace solver {
data_base::data_base(const ocp_ptr_t &prob)
    : node_data(prob), nx(prob->dim(__x)), nu(prob->dim(__u)),
      Q_x(dense_->jac_[__x]), Q_u(dense_->jac_[__u]),
      Q_y(dense_->jac_[__y]), Q_xx(dense_->hessian_[__x][__x]),
      Q_ux(dense_->hessian_[__u][__x]), Q_uu(dense_->hessian_[__u][__u]),
      Q_yx(dense_->hessian_[__y][__x]), Q_yy(dense_->hessian_[__y][__y]) {
    prim_step[__x].resize(nx);
    prim_step[__x].setZero();
    prim_step[__u].resize(nu);
    prim_step[__y].resize(nx);
    prim_corr[__x].resize(nx);
    prim_corr[__x].setZero();
    prim_corr[__u].resize(nu);
    prim_corr[__y].resize(nx);
    Q_y_corr = nullptr;
}
} // namespace solver
} // namespace moto
