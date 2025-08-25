#include <moto/ocp/problem.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
namespace moto {
namespace solver {
ipm_constr::approx_data::approx_data(base::approx_data &&rhs)
    : base::approx_data(std::move(rhs)) {
    slack_.resize(func_.dim());
    slack_.setConstant(1e-6);
    diag_scaling.resize(func_.dim());
    diag_scaling.setZero();
    scaled_res_.resize(func_.dim());
    scaled_res_.setZero();
    reg_.resize(func_.dim());
    reg_.setZero();
    reg_T_inv_.resize(func_.dim());
    reg_T_inv_.setZero();
    active_.resize(func_.dim());
    active_.setZero();
}
void ipm_constr::initialize(ipm::data_map_t &data) const {
    // dont call value_impl here
    auto &d = data.as<ipm_data>();
    // dont do d.g_ = d.v_; cuz already set in value_impl
    d.slack_ = (-d.g_).cwiseMax(1e-2); // clip
    d.v_ = d.g_ + d.slack_;            // r_g = g_ + slack
    d.multiplier_.array() = d.ipm_cfg->mu / d.slack_.array();
    d.multiplier_ = d.multiplier_.cwiseMin(1e3); // clip
    d.r_s_.array() = d.multiplier_.cwiseProduct(d.slack_).array();
    if (d.multiplier_.hasNaN() || d.slack_.hasNaN()) {
        fmt::print("multiplier: {}\n", d.multiplier_);
        fmt::print("slack: {}\n", d.slack_);
        throw std::runtime_error("ipm_constr initialization failed due to NaN");
    }
}
void ipm_constr::finalize_newton_step(ipm::data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    size_t arg_idx = 0;
    // update slack newton step
    d.d_slack_ = -d.v_; // -r_g
    // compute linear step
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim) {
            d.d_slack_.noalias() -= d.jac_[arg_idx] * d.prim_step_[arg_idx];
        }
        arg_idx++;
    }
    // update dual newton step
    if (d.ipm_cfg->ipm_computing_affine_step())
        d.d_multiplier_.array() = -(d.r_s_.array() + d.multiplier_.array() * d.d_slack_.array()) / d.reg_T_inv_.array();
    //     d.d_multiplier_.array() = -d.multiplier_.array() - d.diag_scaling.array() * d.d_slack_.array();
    else
        d.d_multiplier_.array() = -(d.r_s_.array() - d.ipm_cfg->mu + d.multiplier_.array() * d.d_slack_.array()) / d.reg_T_inv_.array();
    // d.d_multiplier_.array() = -d.multiplier_.array() + d.ipm_cfg->mu / (d.slack_.array() + d.reg_) - d.diag_scaling.array() * d.d_slack_.array();
    d.d_slack_.array() += d.reg_.array() * d.d_multiplier_.array();
    d.d_slack_.array() *= d.active_.array();      // apply the active set
    d.d_multiplier_.array() *= d.active_.array(); // apply the active
    // if (!d.ipm_cfg->ipm_computing_affine_step()) {
    //     fmt::print("diag_scaling: {:.3}, {:.3}\n", d.diag_scaling.transpose(), (d.active_.array() * d.multiplier_.array() / d.slack_.array()).matrix().transpose());
    //     fmt::print("ipm_constr: d_slack: {}, d_multiplier: {}\n", d.d_slack_.transpose(), d.d_multiplier_.transpose());
    // }
}
void ipm_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    if (d.ipm_cfg->ipm_accept_corrector()) { // add the dual correction term
        d.scaled_res_.array() = d.ipm_cfg->mu - d.corrector_.array();
    } else {
        d.scaled_res_.array() = d.ipm_cfg->mu;
    }
    d.scaled_res_.array() /= (d.slack_.array() + d.reg_.array() * d.multiplier_.array());
    d.scaled_res_.array() *= d.active_.array();
    propagate_jacobian(d);
}
void ipm_constr::update_linesearch_bounds(ipm::data_map_t &data, workspace_data *cfg) const {
    constexpr scalar_t tau = 0.995; // scaling factor
    scalar_t alpha_max = 1.0;       // default max step size
    auto &d = data.as<ipm_data>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    // compute alpha_max
    for (size_t idx : range(dim_)) {
        if (d.d_slack_(idx) < 0 and d.active_(idx) > 0) {
            alpha_max = (-tau) * d.slack_(idx) / d.d_slack_(idx);
            ls_cfg.primal.clip(alpha_max);
            ls_cfg.primal.alpha_max = alpha_max;
        }
        if (d.d_multiplier_(idx) < 0 and d.active_(idx) > 0) {
            alpha_max = (-tau) * d.multiplier_(idx) / d.d_multiplier_(idx);
            ls_cfg.dual.clip(alpha_max);
            ls_cfg.dual.alpha_max = alpha_max;
        }
    }
    // ls_cfg.primal.alpha_max = std::max(ls_cfg.primal.alpha_max, d.reg_);
    // ls_cfg.dual.alpha_max = std::m ax(ls_cfg.dual.alpha_max, d.reg_);
    assert(ls_cfg.primal.alpha_max >= 0);
    assert(ls_cfg.dual.alpha_max > 1e-20);
}
void ipm_constr::finalize_predictor_step(ipm::data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<ipm_data>();
    auto &ipm_worker = cfg->as<ipm_config::worker_type>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    // if we are in the affine step mode, we need to update the ipm worker data
    ipm_worker.n_ipm_cstr += (d.active_.array() > 0).count();
    ipm_worker.prev_aff_comp += d.multiplier_.dot(d.slack_.cwiseProduct(d.active_));
    // finalize the affine step
    d.corrector_.array() = ls_cfg.alpha_dual * d.d_multiplier_.array() * ls_cfg.alpha_primal * d.d_slack_.array();
    d.d_multiplier_.array() *= ls_cfg.alpha_dual;
    d.d_slack_.array() *= ls_cfg.alpha_primal;
    ipm_worker.post_aff_comp += (d.multiplier_ + d.d_multiplier_).cwiseProduct(d.active_).dot(d.slack_ + d.d_slack_);
    // assert(d.multiplier_.dot(d.slack_) > 0 &&
    //        "the complementarity must be positive before the line search step");
    // assert((d.multiplier_ + d.d_multiplier_).dot(d.slack_ + d.d_slack_) > 0);
}
void ipm_constr::apply_affine_step(ipm::data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<ipm_data>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    assert(!d.ipm_cfg->ipm_computing_affine_step() && "ipm affine step computation not ended");
    d.d_slack_.array() *= ls_cfg.alpha_primal;
    d.d_multiplier_.array() *= ls_cfg.alpha_dual;
    d.slack_.array() += d.d_slack_.array();
    d.multiplier_.array() += d.d_multiplier_.array();
    // d.slack_.array() += ls_cfg.alpha_primal * d.d_slack_.array();
    // d.multiplier_.array() += ls_cfg.alpha_dual * d.d_multiplier_.array();
    if (d.ipm_cfg->ipm_accept_corrector()) {
        d.slack_ = d.slack_.array().max(1e-20);
        d.multiplier_ = d.multiplier_.array().max(1e-20);
    }
}
void ipm_constr::value_impl(func_approx_data &data) const {
    base::value_impl(data);
    auto &d = data.as<ipm_data>();
    d.g_ = d.v_;            //.cwiseMin(-d.reg_);
    d.v_ = d.g_ + d.slack_; // r_g = g_ + slack
    // for (size_t i = 0; i < dim_; i++) {
    //     if (d.slack_(i) < 1e-8 && d.g_(i) < 1e-8) {
    //         d.reg_(i) = 1e-8; // regularization for slack variables
    //         d.active_[i] = 0; // active constraint
    //         d.slack_(i) = 1e-8;
    //     } else {
    //         d.active_[i] = 0; // inactive constraint
    //         d.reg_(i) = 1e-8; // regularization for slack variables
    //     }
    // }
    d.reg_.setConstant(1e-8);
    d.active_.setConstant(1.0);
    // d.active_.array() = 1 - d.active_.array(); // invert the active set
    d.r_s_.array() = d.multiplier_.cwiseProduct(d.slack_).array();
}
void ipm_constr::jacobian_impl(func_approx_data &data) const {
    base::jacobian_impl(data);
    auto &d = data.as<ipm_data>();

    // setup T^{-1} N
    d.reg_T_inv_.array() = d.slack_.array() + d.reg_.array() * d.multiplier_.array();
    d.diag_scaling.array() = d.active_.array() * d.multiplier_.array() / d.reg_T_inv_.array();

    d.scaled_res_.array() = d.diag_scaling.array() * d.g_.array();
    if (!d.ipm_cfg->ipm_enable_affine_step()) {
        // if we are not in the affine step mode, we need to update the scaled residual with mu
        d.scaled_res_.array() += d.ipm_cfg->mu / d.reg_T_inv_.array();
    }
    d.scaled_res_.array() *= d.active_.array(); // only active constraints contribute to the scaled residual
    propagate_jacobian(d);
    propagate_hessian(d);
}
void ipm_constr::propagate_jacobian(func_approx_data &data) const {
    size_t j_idx = 0;
    bool nan_found = false;
    auto &d = data.as<ipm_data>();
    for (auto &j : d.jac_) {
        if (j.size() != 0) {
            d.jac_modification_[j_idx].noalias() += d.scaled_res_.transpose() * j;
            if (d.jac_modification_[j_idx].hasNaN()) {
                nan_found = true;
                fmt::print("--------------------\n");
                fmt::print("constraint name: {}\n", d.func_.name());
                for (sym &arg : d.func_.in_args()) {
                    fmt::print("arg: {}: {}\n", arg.name(), d[arg].transpose());
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
    if (nan_found) {
        fmt::print("NaN found in jacobian modification for constraint: {}\n", d.func_.name());
        throw std::runtime_error("NaN found in jacobian modification");
    }
}
void ipm_constr::propagate_hessian(func_approx_data &d) const {
    // modification of hessian
    size_t outer_idx = 0;
    for (auto &outer : d.merit_hess_) {
        size_t inner_idx = 0;
        if (outer.size()) { // skip empty hess
            for (auto &inner : outer) {
                if (inner.size() != 0) {
                    inner.noalias() += d.jac_[outer_idx].transpose() * d.as<ipm_data>().diag_scaling.asDiagonal() * d.jac_[inner_idx];
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
} // namespace solver
} // namespace moto
