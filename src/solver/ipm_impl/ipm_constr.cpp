#include <moto/solver/ipm_constr.hpp>
#include <moto/solver/line_search.hpp>

namespace moto {
namespace ipm_impl {
void ipm_constr_impl::initialize(soft_constr_data &data) {
    constr_impl::value_impl(data);
    auto &d = static_cast<ipm_data &>(data);
    d.slack_ = (-data.v_).cwiseMax(1e-2); // clip
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
    // compute linear step
    for (const auto &arg : d.func_.in_args()) {
        if (arg->field_ < field::num_prim) {
            d.d_slack_.noalias() -= d.jac_[arg_idx] * d.prim_step_[arg_idx];
        }
        arg_idx++;
    }
    // d.d_slack_ = d.d_slack_.array().max(1e-8 - d.slack_.array());
    // update dual newton step
    d.d_multipler_.array() = -d.multiplier_.array() + d.mu_ / d.slack_.array() - d.diag_scaling.array() * d.d_slack_.array();
    /// @todo iterative refinement for better accuracy?
}
void ipm_constr_impl::update_line_search_cfg(soft_constr_data &data, solver::line_search_cfg *cfg) {
    constexpr scalar_t tau = 0.995; // scaling factor
    scalar_t alpha_max = 1.0;       // default max step size
    auto &d = static_cast<ipm_data &>(data);
    // compute alpha_max
    for (size_t idx : range(dim_)) {
        if (d.d_slack_(idx) < 0) {
            alpha_max = (-tau) * d.slack_(idx) / d.d_slack_(idx);
            cfg->primal.clip(alpha_max);
            cfg->primal.alpha_max = alpha_max;
        }
        if (d.d_multipler_(idx) < 0) {
            alpha_max = (-tau) * d.multiplier_(idx) / d.d_multipler_(idx);
            cfg->dual.clip(alpha_max);
            cfg->dual.alpha_max = alpha_max;
        }
    }
}
void ipm_constr_impl::line_search_step(soft_constr_data &data, solver::line_search_cfg *cfg) {
    auto &d = static_cast<ipm_data &>(data);
    d.slack_.array() += cfg->alpha_primal * d.d_slack_.array();
    d.multiplier_.array() += cfg->alpha_dual * d.d_multipler_.array();
    fmt::print("-----------line search---------\n");
    fmt::print("constraint name: {}\n", d.func_.name_);
    size_t arg_idx = 0;
    for (auto &arg : d.func_.in_args()) {
        fmt::print("arg: {}: {}\n", arg->name_, d.prim_step_[arg_idx].transpose());
        arg_idx++;
    }
    fmt::print("v: {:.3}\n", d.v_.transpose());
    fmt::print("d_slack: {:.3}\n", d.d_slack_.transpose());
    fmt::print("slack: {:.3}\n", d.slack_.transpose());
    fmt::print("multiplier: {:.3}\n", d.multiplier_.transpose());
    d.comp_res_.array() = d.multiplier_.cwiseProduct(d.slack_).array() - d.mu_;
    fmt::print("d.comp_res_: {:.3}\n", d.comp_res_.transpose());
    fmt::print("scaling : {:.3}\n", d.multiplier_.cwiseQuotient(d.slack_));
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