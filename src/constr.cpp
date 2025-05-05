#include <atri/ocp/constr.hpp>
#include <atri/ocp/core/problem_data.hpp>

namespace atri {
constr_data::constr_data(problem_data *raw,
                         sparse_approx_data &&d,
                         constr *f)
    : sparse_approx_data(std::move(d)),
      multiplier_(raw->dual_[f->field_].segment(raw->prob_->get_expr_start(*f),
                                                f->dim_)) {
    const auto &in_args = f->in_args();
    for (size_t i = 0; i < in_args_.size(); i++) {
        vjp_.push_back(raw->jac_[in_args[i]->field_].segment(
            raw->prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
    }
}
void constr::jacobian_impl(sparse_approx_data_ptr_t data) {
    // compute jacobian first
    jacobian(data);
    // update multiplier - jacobian product
    auto d = std::static_pointer_cast<constr_data>(data);
    for (size_t i = 0; i < data->in_args_.size(); i++) {
        d->vjp_[i].noalias() += d->multiplier_.transpose() * d->jac_[i];
    }
}
void constr::hessian_impl(sparse_approx_data_ptr_t data) {
    // do not compute the whole hessian
    // preferred: first do multipler.T * jac then use auto-differentiation
    hessian(std::static_pointer_cast<constr_data>(data));
}
} // namespace atri
