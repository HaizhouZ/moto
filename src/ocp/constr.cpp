#include <atri/core/external_function.hpp>
#include <atri/ocp/approx_storage.hpp>
#include <atri/ocp/constr.hpp>
#include <iostream>
namespace atri {
constr_data::constr_data(approx_storage *raw,
                         sparse_approx_data &&d,
                         constr_impl *f)
    : sparse_approx_data(std::move(d)), merit_(&raw->cost_),
      multiplier_(raw->dual_[f->field_].segment(raw->prob_->get_expr_start(*f),
                                                f->dim_)) {
    const auto &in_args = f->in_args();
    for (size_t i = 0; i < in_args_.size(); i++) {
        vjp_.push_back(raw->jac_[in_args[i]->field_].segment(
            raw->prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
    }
}

void constr_impl::load_external(const std::string &path) {
    auto funcs = load_approx(name_, true, order() >= approx_order::first, order() >= approx_order::second);
    value = [eval = funcs[0]](sparse_approx_data &d) {
        eval.invoke(d.in_args_, d.v_);
    };
    jacobian = [jac = funcs[1]](sparse_approx_data &d) {
        jac.invoke(d.in_args_, d.jac_);
    };

    hessian = [hess = funcs[2]](constr_data &d) {
        hess.invoke(d.in_args_, d.hess_);
    };
}
void constr_impl::value_impl(sparse_approx_data &data) {
    value(data);
    // compute contribution to merit function
    auto &d = static_cast<constr_data &>(data);
    *d.merit_ += d.multiplier_.dot(d.v_);
    // fmt::print("\t{}:\tv:{}\n", name_, d.v_.transpose());
    // fmt::print("\t{}:\tm:{}\n", name_, d.multiplier_.transpose());
}
void constr_impl::jacobian_impl(sparse_approx_data &data) {
    // compute jacobian first
    jacobian(data);
    // update multiplier - jacobian product
    auto &d = static_cast<constr_data &>(data);
    for (size_t i = 0; i < d.in_args_.size(); i++) {
        // fmt::print("{}\t{}:i\t{:.3}\n", i, name_, d.in_args_[i].transpose());
        d.vjp_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
        // fmt::print("{}\t{}:jac\n{:.3}\n", i, name_, d.jac_[i]);
    }
}
void constr_impl::hessian_impl(sparse_approx_data &data) {
    // do not compute the whole hessian
    // preferred: first do multipler.T * jac then use auto-differentiation
    hessian(static_cast<constr_data &>(data));
}
} // namespace atri
