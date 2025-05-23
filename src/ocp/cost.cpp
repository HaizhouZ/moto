#include <atri/core/external_function.hpp>
#include <atri/ocp/cost.hpp>

namespace atri {
void cost_impl::load_external(const std::string &path) {
    auto funcs = load_approx(name_, true, order() >= approx_order::first, order() >= approx_order::second);
    value = [eval = funcs[0]](sparse_approx_data &d) {
        eval.invoke(d.in_args_, d.v_);
    };
    jacobian = [jac = funcs[1]](sparse_approx_data &d) {
        jac.invoke(d.in_args_, d.jac_);
    };

    hessian = [hess = funcs[2]](sparse_approx_data &d) {
        hess.invoke(d.in_args_, d.hess_);
    };
}
} // namespace atri