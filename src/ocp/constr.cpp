#include <atri/core/external_function.hpp>
#include <atri/ocp/approx_storage.hpp>
#include <atri/ocp/constr.hpp>
#include <filesystem>
#include <iostream>
namespace atri {
constr_data::constr_data(approx_storage *raw,
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
void constr_impl::load_external(const std::string &path) {
    std::filesystem::path p(path);
    ext_func eval(p / ("lib" + name_ + ".so"), name_);
    value = [eval](sparse_approx_data &d) {
        eval.invoke(d.in_args_, d.v_);
    };
    if (order() >= approx_order::first) {
        ext_func jac(p / ("lib" + name_ + "_jac.so"), name_ + "_jac");
        jacobian = [jac](sparse_approx_data &d) {
            jac.invoke(d.in_args_, d.jac_);
        };
    }
    if (order() >= approx_order::second) {
        ext_func hess(p / ("lib" + name_ + "_hess.so"), name_ + "_hess");
        hessian = [hess](constr_data &d) {
            hess.invoke(d.in_args_, d.jac_);
        };
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
