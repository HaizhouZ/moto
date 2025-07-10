#include <moto/solver/ipm_constr.hpp>

namespace moto {
namespace ipm_impl {
void ipm_constr_impl::initialize(soft_constr_data &data) {
    constr_impl::value_impl(data);
    auto &d = static_cast<ipm_data &>(data);
    d.slack_ = (-data.v_).cwiseMax(1e-3); // clip
    d.mu_ = 1e-2;
    d.multiplier_.array() = d.mu_ / d.slack_.array();
    d.multiplier_ = d.multiplier_.cwiseMin(1e3); // clip
}
void ipm_constr_impl::post_rollout(soft_constr_data &data) {
    auto &d = static_cast<ipm_data &>(data);
    size_t arg_idx = 0;
    // update slack newton step
    d.d_slack_ = -(d.v_ + d.slack_); // +r_g
    // ensure slack + step >= 1e-8
    d.d_slack_ = d.d_slack_.array().max(1e-8 - d.slack_.array());
    // compute linear step
    for (const auto &arg : d.func_.in_args()) {
        if (arg->field_ < field::num_prim) {
            d.d_slack_.noalias() -= d.jac_[arg_idx] * d.prim_step_[arg_idx];
        }
        arg_idx++;
    }
    // update dual newton step
    d.d_multipler_.array() = -d.multiplier_.array() + d.mu_ / d.slack_.array() - d.diag_scaling.array() * d.d_slack_.array();
    /// @todo iterative refinement for better accuracy?
}
void ipm_constr_impl::line_search_step(soft_constr_data &data, scalar_t alpha) {
    auto &d = static_cast<ipm_data &>(data);
    d.slack_.array() += alpha * d.d_slack_.array();
    d.multiplier_.array() *= alpha * d.d_multipler_.array();
    // ensure slack + step >= 1e-8
    d.slack_ = d.slack_.array().max(1e-8);
}
void ipm_constr_impl::value_impl(sp_approx_map &data) {
    constr_impl::value_impl(data);
    auto &d = static_cast<ipm_data &>(data);
    d.comp_res_.array() = d.multiplier_.cwiseProduct(d.slack_).array() - d.mu_;
}
void ipm_constr_impl::jacobian_impl(sp_approx_map &data) {
    constr_impl::jacobian_impl(data);
    auto &d = static_cast<ipm_data &>(data);
    // setup T^{-1} N
    d.diag_scaling.array() = d.multiplier_.array() / d.slack_.array();
    // set scaled residual
    d.scaled_res_ = d.diag_scaling.cwiseProduct(d.v_);
    d.scaled_res_.array() += d.mu_ / d.slack_.array();
    // modification of jacobian
    // fmt::print("--------------------\n");
    // fmt::print("constraint name: {}\n", d.func_.name_);
    // for (auto &arg : d.func_.in_args()) {
    //     fmt::print("arg: {}: {}\n", arg->name_, d[arg].transpose());
    // }
    size_t j_idx = 0;
    for (auto &j : d.jac_) {
        if (j.rows() != 0 && j.cols() != 0) { // skip empty jac
                                              // if (d.vjp_[j_idx].hasNaN()) {
            // fmt::print("scaled_res: {:.3}\n", d.scaled_res_.transpose());
            // fmt::print("NaN in vjp[{}]\n", j_idx);
            // }
            d.vjp_[j_idx].noalias() += d.scaled_res_.transpose() * j;
            // fmt::print("vjp: {:.3}\n", d.vjp_[j_idx]);
            if (d.vjp_[j_idx].hasNaN()) {
                fmt::print("--------------------\n");
                fmt::print("constraint name: {}\n", d.func_.name_);
                for (auto &arg : d.func_.in_args()) {
                    fmt::print("arg: {}: {}\n", arg->name_, d[arg].transpose());
                }
                fmt::print("jac: \n{:.3}\n", j);
                fmt::print("slack: {:.3}\n", d.slack_.transpose());
                fmt::print("multiplier: {:.3}\n", d.multiplier_.transpose());
                fmt::print("diag_scaling: {:.3}\n", d.diag_scaling.transpose());
                fmt::print("scaled_res: {:.3}\n", d.scaled_res_.transpose());
                fmt::print("vjp: {:.3}\n", d.vjp_[j_idx]);
                fmt::print("NaN in vjp[{}]\n", j_idx);
            }
        }
        j_idx++;
    }
    // modification of hessian
    size_t outer_idx = 0;
    for (auto &outer : d.hess_) {
        size_t inner_idx = 0;
        if (outer.size()) { // skip empty hess
            for (auto &inner : outer) {
                if (inner.rows() != 0 && inner.cols() != 0) {
                    inner.noalias() += d.jac_[outer_idx].transpose() * d.diag_scaling.asDiagonal() * d.jac_[inner_idx];
                    if (inner.hasNaN() || d.diag_scaling.hasNaN() || d.jac_[inner_idx].hasNaN() || d.jac_[outer_idx].hasNaN()) {
                        fmt::print("NaN in hess[{}][{}]\n", outer_idx, inner_idx);
                    }
                }
                inner_idx++;
            }
        }
        outer_idx++;
    }
}
} // namespace ipm_impl
} // namespace moto