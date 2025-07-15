#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/data_base.hpp>

namespace moto {
namespace solver {
data_base::data_base(const ocp_ptr_t &prob)
    : node_data(prob), nx(prob->dim_[__x]), nu(prob->dim_[__u]),
      Q_x(dense_->jac_[__x]), Q_u(dense_->jac_[__u]),
      Q_y(dense_->jac_[__y]), Q_xx(dense_->hessian_[__x][__x]),
      Q_ux(dense_->hessian_[__u][__x]), Q_uu(dense_->hessian_[__u][__u]),
      Q_yx(dense_->hessian_[__y][__x]), Q_yy(dense_->hessian_[__y][__y]) {
    prim_step[__x].resize(nx);
    prim_step[__x].setZero();
    prim_step[__u].resize(nu);
    prim_step[__y].resize(nx);
    // initialize soft constraint data
    for (auto f : {__ineq_x, __ineq_xu, __eq_x_soft, __eq_xu_soft}) {
        for (auto &d : sparse_[f]) {
            auto &sd = static_cast<impl::soft_constr_data &>(*d);
            sd.prim_step_.clear();
            for (const auto &arg : sd.func_.in_args()) {
                if (arg->field_ < field::num_prim) {
                    sd.prim_step_.push_back(prob->extract(prim_step[arg->field_], *arg));
                }
            }
        }
    }
}
} // namespace solver
} // namespace moto
