#include <atri/ocp/core/approx_data.hpp>
#include <atri/ocp/func/constr.hpp>

namespace atri {
constr_data::constr_data(approx_data *raw,
                         sparse_approx_data &&d,
                         constr_impl *f)
    : sparse_approx_data(std::move(d)),
      multiplier_(raw->dual_[f->field_].segment(raw->prob_->get_expr_start(*f),
                                                f->dim_)) {
    const auto &in_args = f->in_args();
    for (size_t i = 0; i < in_args_.size(); i++) {
        vjp_.push_back(raw->jac_[in_args[i]->field_].segment(
            raw->prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
    }
}
void constr_impl::jacobian_impl(sparse_approx_data &data) {
    // compute jacobian first
    jacobian(data);
    // update multiplier - jacobian product
    auto &d = static_cast<constr_data &>(data);
    for (size_t i = 0; i < d.in_args_.size(); i++) {
        d.vjp_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
    }
}
void constr_impl::hessian_impl(sparse_approx_data &data) {
    // do not compute the whole hessian
    // preferred: first do multipler.T * jac then use auto-differentiation
    hessian(static_cast<constr_data &>(data));
}
} // namespace atri
