#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto {
namespace ipm_impl {
void ipm_constr::initialize(ipm::data_map_t &data) {
    base::value_impl(data);
    auto &d = data.as<ipm_data>();
    d.g_ = d.v_;
    d.slack_ = (-d.g_).cwiseMax(1e-2); // clip
    d.v_ = d.g_ + d.slack_;            // r_g = g_ + slack
    d.multiplier_.array() = d.ipm_cfg->mu / d.slack_.array();
    d.multiplier_ = d.multiplier_.cwiseMin(1e3); // clip
    d.r_s_.array() = d.multiplier_.cwiseProduct(d.slack_).array() - d.ipm_cfg->mu;
}
void ipm_constr::finalize_newton_step(ipm::data_map_t &data) {
    auto &d = data.as<ipm_data>();
    size_t arg_idx = 0;
    // update slack newton step
    d.d_slack_ = -d.v_; // -r_g
    // ensure slack + step >= 1e-8
    // compute linear step
    for (const auto &arg : d.func_.in_args()) {
        if (arg->field_ < field::num_prim) {
            d.d_slack_.noalias() -= d.jac_data_[arg_idx] * d.prim_step_[arg_idx];
        }
        arg_idx++;
    }
    // update dual newton step
    if (d.ipm_cfg->ipm_computing_affine_step())
        d.d_multipler_.array() = -d.multiplier_.array() - d.diag_scaling.array() * d.d_slack_.array();
    else
        d.d_multipler_.array() = -d.multiplier_.array() + d.ipm_cfg->mu / d.slack_.array() - d.diag_scaling.array() * d.d_slack_.array();
}
void ipm_constr::correct_jacobian(data_map_t &data) {
    auto &d = data.as<ipm_data>();
    if (d.ipm_cfg->ipm_accept_corrector()) { // add the dual correction term
        d.scaled_res_.array() = d.ipm_cfg->mu - d.d_multipler_.array() * d.d_slack_.array();
    } else {
        d.scaled_res_.array() = d.ipm_cfg->mu;
    }
    d.scaled_res_.array() /= d.slack_.array();
    propagate_jacobian(d);
}
void ipm_constr::update_linesearch_config(ipm::data_map_t &data, workspace_data *cfg) {
    constexpr scalar_t tau = 0.995; // scaling factor
    scalar_t alpha_max = 1.0;       // default max step size
    auto &d = data.as<ipm_data>();
    auto &ls_cfg = cfg->get<solver::linesearch_config>();
    // compute alpha_max
    for (size_t idx : range(dim_)) {
        if (d.d_slack_(idx) < 0) {
            alpha_max = (-tau) * d.slack_(idx) / d.d_slack_(idx);
            ls_cfg.primal.clip(alpha_max);
            ls_cfg.primal.alpha_max = alpha_max;
        }
        if (d.d_multipler_(idx) < 0) {
            alpha_max = (-tau) * d.multiplier_(idx) / d.d_multipler_(idx);
            ls_cfg.dual.clip(alpha_max);
            ls_cfg.dual.alpha_max = alpha_max;
        }
    }
    assert(ls_cfg.primal.alpha_max > 1e-3);
    assert(ls_cfg.dual.alpha_max > 1e-3);
}
void ipm_constr::line_search_step(ipm::data_map_t &data, workspace_data *cfg) {
    auto &d = data.as<ipm_data>();
    auto *ipm_worker = cfg->try_get<ipm_config::worker_type>();
    auto &ls_cfg = cfg->get<solver::linesearch_config>();
    if (ipm_worker && d.ipm_cfg->ipm_enable_affine_step()) {
        assert(d.ipm_cfg->ipm_computing_affine_step() &&
               "ipm affine step computation not started but affine step is requested");
        // if we are in the affine step mode, we need to update the ipm worker data
        ipm_worker->n_ipm_cstr += dim_;
        ipm_worker->prev_aff_comp += d.multiplier_.dot(d.slack_);
        ipm_worker->post_aff_comp += (d.multiplier_ + ls_cfg.alpha_dual * d.d_multipler_)
                                         .dot(d.slack_ + ls_cfg.alpha_primal * d.d_slack_);
        assert(d.multiplier_.dot(d.slack_) > 0 &&
               "the complementarity must be positive before the line search step");
        assert((d.multiplier_ + ls_cfg.alpha_dual * d.d_multipler_)
                   .dot(d.slack_ + ls_cfg.alpha_primal * d.d_slack_) > 0);
    } else {
        assert(!d.ipm_cfg->ipm_computing_affine_step() &&
               "ipm affine step computation not ended");
        d.slack_.array() += ls_cfg.alpha_primal * d.d_slack_.array();
        d.multiplier_.array() += ls_cfg.alpha_dual * d.d_multipler_.array();
        if (d.ipm_cfg->ipm_enable_corrector()) {
            d.slack_ = d.slack_.array().max(1e-20);
            d.multiplier_ = d.multiplier_.array().max(1e-20);
        }
    }
}
void ipm_constr::value_impl(sp_approx_map &data) {
    base::value_impl(data);
    auto &d = data.as<ipm_data>();
    d.g_ = d.v_;
    d.v_ = d.g_ + d.slack_; // r_g = g_ + slack
    d.r_s_.array() = d.multiplier_.cwiseProduct(d.slack_).array() - d.ipm_cfg->mu;
}
void ipm_constr::jacobian_impl(sp_approx_map &data) {
    base::jacobian_impl(data);
    auto &d = data.as<ipm_data>();
    // setup T^{-1} N
    d.diag_scaling.array() = d.multiplier_.array() / d.slack_.array();
    // set scaled residual
    d.scaled_res_ = d.diag_scaling.cwiseProduct(d.g_);
    if (!d.ipm_cfg->ipm_enable_affine_step())
        // if we are not in the affine step mode, we need to update the scaled residual with mu
        d.scaled_res_.array() += d.ipm_cfg->mu / d.slack_.array();
    // modification of jacobian
    // fmt::print("--------------------\n");
    // fmt::print("constraint name: {}\n", d.func_.name_);
    // for (auto &arg : d.func_.in_args()) {
    //     fmt::print("arg: {}: {}\n", arg->name_, d[arg].transpose());
    // }
    propagate_jacobian(d);
    propagate_hessian(d);
}
void ipm_constr::propagate_jacobian(ipm_data &d) {
    size_t j_idx = 0;
    for (auto &j : d.jac_data_) {
        if (j.size() != 0) {
            d.jac_modification_[j_idx].noalias() += d.scaled_res_.transpose() * j;
            if (d.jac_modification_[j_idx].hasNaN()) {
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
                fmt::print("jac modification: {:.3}\n", d.jac_modification_[j_idx]);
                fmt::print("NaN in jac modification[{}]\n", j_idx);
            }
        }
        j_idx++;
    }
}
void ipm_constr::propagate_hessian(ipm_data &d) {
    // modification of hessian
    size_t outer_idx = 0;
    for (auto &outer : d.hess_) {
        size_t inner_idx = 0;
        if (outer.size()) { // skip empty hess
            for (auto &inner : outer) {
                if (inner.size() != 0) {
                    inner.noalias() += d.jac_data_[outer_idx].transpose() * d.diag_scaling.asDiagonal() * d.jac_data_[inner_idx];
                    if (inner.hasNaN()) {
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