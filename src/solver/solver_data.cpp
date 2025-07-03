#include <moto/solver/solver_data.hpp>

namespace moto {
namespace solver {
solver_data::solver_data(const ocp_ptr_t &prob, approx_storage *dense_)
    : nx(prob->dim_[__x]), nu(prob->dim_[__u]),
      Q_x(dense_->jac_[__x]), Q_u(dense_->jac_[__u]),
      Q_y(dense_->jac_[__y]), Q_xx(dense_->hessian_[__x][__x]),
      Q_ux(dense_->hessian_[__u][__x]), Q_uu(dense_->hessian_[__u][__u]),
      Q_yx(dense_->hessian_[__y][__x]), Q_yy(dense_->hessian_[__y][__y]) {
    prim_rollout_[__x].resize(nx);
    prim_rollout_[__x].setZero();
    prim_rollout_[__u].resize(nu);
    prim_rollout_[__y].resize(nx);
}
} // namespace solver
} // namespace moto
