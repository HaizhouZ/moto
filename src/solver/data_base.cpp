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
    prim_corr[__x].resize(nx);
    prim_corr[__x].setZero();
    prim_corr[__u].resize(nu);
    prim_corr[__y].resize(nx);
    Q_y_cache.resize(Q_y.size());
    // initialize soft constraint data
    for (auto f : concat_fields(ineq_constr_fields, soft_constr_fields)) {
        for (auto &d : sparse_[f]) {
            auto &sd = dynamic_cast<impl::soft_constr::sp_data_map &>(*d);
            sd.prim_step_.clear();
            for (const auto &arg : sd.func_.in_args()) {
                if (arg->field_ < field::num_prim) {
                    sd.prim_step_.push_back(prob->extract(prim_step[arg->field_], *arg));
                }
            }
        }
    }
}
void prepare_correction(data_base *data) {
    data->Q_y_cache = data->Q_y; // cache the Q_y before correction
    data->Q_x.setZero();
    data->Q_u.setZero();
    data->Q_y.setZero();
    data->prim_corr[__x].setZero();
}
} // namespace solver
} // namespace moto
