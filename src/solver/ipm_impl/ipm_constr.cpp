#include <moto/ocp/problem.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
namespace moto {
namespace solver {

namespace {
using ipm_data = ipm_constr::ipm_data;
using ipm_side_data = ipm_constr::approx_data::side_data;
using box_pair_runtime = ineq_constr::approx_data::box_pair_runtime;
using box_mask = Eigen::Array<bool, Eigen::Dynamic, 1>;

template <typename Fn>
void for_each_box_side(ipm_data &d, const ineq_constr::box_spec &box, Fn &&fn) {
    for (auto side : box_sides) if (box.has_side[side]) {
        auto &pair = *d.box_side_[side];
        fn(side, pair, static_cast<ipm_side_data &>(pair), box.present_mask[side]);
    }
}

template <typename Fn>
void for_each_box_side(const ipm_data &d, const ineq_constr::box_spec &box, Fn &&fn) {
    for (auto side : box_sides) if (box.has_side[side]) {
        const auto &pair = *d.box_side_[side];
        fn(side, pair, static_cast<const ipm_side_data &>(pair), box.present_mask[side]);
    }
}

void refresh_box_state(ipm_constr::approx_data &d) {
    const auto &box = d.require_box_spec("refresh_box_state");
    d.lifted_view.setConstant(d.v_.size(), -std::numeric_limits<scalar_t>::infinity());
    d.multiplier_.setZero();
    d.d_multiplier_.setZero();
    d.comp_.setZero();
    for_each_box_side(d, box, [&](auto side, box_pair_runtime &pair, ipm_side_data &ipm_side, const box_mask &present) {
        ipm_side.r_s.array() = present.select(pair.multiplier.array() * pair.slack.array(), scalar_t(0));
        d.lifted_view.array() =
            d.lifted_view.array().max(present.select(pair.residual.array() + pair.slack.array(),
                                                     -std::numeric_limits<scalar_t>::infinity()));
        const scalar_t dual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
        d.multiplier_.array() += dual_sign * present.select(pair.multiplier.array(), scalar_t(0));
        d.d_multiplier_.array() += dual_sign * present.select(pair.d_multiplier.array(), scalar_t(0));
        d.comp_.array() =
            (ipm_side.r_s.array().abs() >= d.comp_.array().abs()).select(ipm_side.r_s.array(), d.comp_.array());
    });
    d.v_ = d.lifted_view;
}

void compute_side_reg_scaling(const ineq_constr::approx_data::box_pair_runtime &pair,
                              ipm_constr::approx_data::side_data &side,
                              const Eigen::Array<bool, Eigen::Dynamic, 1> &present,
                              scalar_t mu,
                              bool include_barrier_shift) {
    side.reg_T_inv.array() = present.select(pair.slack.array() + side.reg.array() * pair.multiplier.array(), scalar_t(0));
    const auto denom = present.select(side.reg_T_inv.array(), scalar_t(1));
    side.diag_scaling.array() = present.select(pair.multiplier.array() / denom, scalar_t(0));
    side.scaled_res.array() = present.select(side.diag_scaling.array() * pair.residual.array(), scalar_t(0));
    if (include_barrier_shift) {
        side.scaled_res.array() += present.select(mu / denom, scalar_t(0));
    }
}

void compute_side_corrector_scaling(const ineq_constr::approx_data::box_pair_runtime &pair,
                                    ipm_constr::approx_data::side_data &side,
                                    const Eigen::Array<bool, Eigen::Dynamic, 1> &present,
                                    scalar_t mu,
                                    bool use_corrector) {
    side.reg_T_inv.array() = present.select(pair.slack.array() + side.reg.array() * pair.multiplier.array(), scalar_t(0));
    const auto denom = present.select(side.reg_T_inv.array(), scalar_t(1));
    if (use_corrector) {
        side.scaled_res.array() = present.select((scalar_t(mu) - side.corrector.array()) / denom, scalar_t(0));
    } else {
        side.scaled_res.array() = present.select(mu / denom, scalar_t(0));
    }
}

void compute_side_newton_step(ineq_constr::approx_data::box_pair_runtime &pair,
                              ipm_constr::approx_data::side_data &side,
                              const Eigen::Array<bool, Eigen::Dynamic, 1> &present,
                              scalar_t mu,
                              bool affine_step) {
    const auto denom = present.select(side.reg_T_inv.array(), scalar_t(1));
    if (affine_step) {
        pair.d_multiplier.array() = present.select(-(side.r_s.array() + pair.multiplier.array() * pair.d_slack.array()) / denom, scalar_t(0));
    } else {
        pair.d_multiplier.array() = present.select(-(side.r_s.array() - mu + pair.multiplier.array() * pair.d_slack.array()) / denom, scalar_t(0));
    }
    pair.d_slack.array() += side.reg.array() * pair.d_multiplier.array();
}
} // namespace

ipm_constr::approx_data::approx_data(base::approx_data &&rhs)
    : base::approx_data(std::move(rhs)) {
    const auto n = v_.size();
    lifted_view.setZero(n);
    jac_step.setZero(n);
}
void ipm_constr::initialize(ipm::data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::initialize");
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
        pair.slack.array() = present.select((-pair.residual.array()).max(scalar_t(1.)), scalar_t(0));
        pair.multiplier.array() = present.template cast<scalar_t>();
        pair.d_slack.setZero();
        pair.d_multiplier.setZero();
    });
    refresh_box_state(d);
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &) {
        pair.slack_backup = pair.slack;
        pair.multiplier_backup = pair.multiplier;
    });
    bool slack_has_nan = false;
    for_each_box_side(d, box, [&](auto, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &) {
        slack_has_nan = slack_has_nan || pair.slack.hasNaN();
    });
    if (d.multiplier_.hasNaN() || slack_has_nan) {
        fmt::print("multiplier: {}\n", d.multiplier_);
        for_each_box_side(d, box, [](auto side, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &) {
            fmt::print("{} slack: {}\n",
                       side == box_side::ub ? "ub" : "lb",
                       pair.slack.transpose());
        });
        throw std::runtime_error("ipm_constr initialization failed due to NaN");
    }
}
void ipm_constr::finalize_newton_step(ipm::data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::finalize_newton_step");
    refresh_box_state(d);
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
        pair.d_slack.array() = present.select(-(pair.residual.array() + pair.slack.array()), scalar_t(0));
    });
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim) {
            if (d.has_jacobian_block(arg_idx)) {
                d.jac_step.noalias() = d.jac_[arg_idx] * d.prim_step_[arg_idx];
                for_each_box_side(d, box, [&](auto side, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
                    const scalar_t slack_sign = side == box_side::ub ? scalar_t(-1) : scalar_t(1);
                    pair.d_slack.array() += slack_sign * present.select(d.jac_step.array(), scalar_t(0));
                });
            }
        }
        arg_idx++;
    }
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &ipm_side, const box_mask &present) {
        compute_side_newton_step(pair,
                                 ipm_side,
                                 present,
                                 d.ipm_cfg->mu,
                                 d.ipm_cfg->ipm_computing_affine_step());
    });
    refresh_box_state(d);
}
void ipm_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::apply_corrector_step");
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &ipm_side, const box_mask &present) {
        compute_side_corrector_scaling(pair,
                                       ipm_side,
                                       present,
                                       d.ipm_cfg->mu,
                                       d.ipm_cfg->ipm_accept_corrector());
    });
    for_each_box_side(d, box, [&](auto side, const box_pair_runtime &pair, const ipm_side_data &ipm_side, const box_mask &) {
        if (ipm_side.scaled_res.hasNaN()) {
            fmt::print("box scaled_res {}: {}\n", side == box_side::ub ? "ub" : "lb", ipm_side.scaled_res.transpose());
            fmt::print("mu : {}, corrector {}: {}\n",
                       d.ipm_cfg->mu,
                       side == box_side::ub ? "ub" : "lb",
                       ipm_side.corrector.transpose());
            fmt::print("slack {}: {}\n",
                       side == box_side::ub ? "ub" : "lb",
                       pair.slack.transpose());
            fmt::print("reg {}: {}\n",
                       side == box_side::ub ? "ub" : "lb",
                       ipm_side.reg.transpose());
            fmt::print("multiplier {}: {}\n",
                       side == box_side::ub ? "ub" : "lb",
                       pair.multiplier.transpose());
            throw std::runtime_error("ipm_constr apply_corrector_step failed due to NaN");
        }
    });
    propagate_jacobian(d);
}
void ipm_constr::update_ls_bounds(ipm::data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<ipm_data>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    const auto &box = d.require_box_spec("ipm_constr::update_ls_bounds");
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &) {
        positivity::update_pair_bounds(ls_cfg, pair.slack, pair.d_slack, pair.multiplier, pair.d_multiplier);
    });
    assert(ls_cfg.primal.alpha_max >= 0);
    assert(ls_cfg.dual.alpha_max > 1e-20);
}
void ipm_constr::backup_trial_state(ipm::data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::backup_trial_state");
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &) {
        positivity::backup_pair(pair.slack, pair.slack_backup, pair.multiplier, pair.multiplier_backup);
    });
}
void ipm_constr::restore_trial_state(ipm::data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::restore_trial_state");
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &) {
        positivity::restore_pair(pair.slack, pair.slack_backup, pair.multiplier, pair.multiplier_backup);
    });
    refresh_box_state(d);
}
void ipm_constr::restoration_commit_dual_step(data_map_t &data, scalar_t alpha_dual) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::restoration_commit_dual_step");
    d.multiplier_.noalias() += alpha_dual * d.d_multiplier_;
    for_each_box_side(d, box, [alpha_dual](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
        pair.multiplier.array() =
            present.select(pair.multiplier.array() + alpha_dual * pair.d_multiplier.array(), scalar_t(0));
        pair.multiplier_backup = pair.multiplier;
    });
}

void ipm_constr::restoration_reset_bound_multipliers(data_map_t &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::restoration_reset_bound_multipliers");
    d.multiplier_.setZero();
    d.d_multiplier_.setZero();
    d.comp_.setZero();
    for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
        pair.multiplier =
            present.select(vector::Ones(pair.multiplier.size()).array(), scalar_t(0)).matrix();
        pair.multiplier_backup = pair.multiplier;
    });
}

scalar_t ipm_constr::search_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const ipm_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("ipm_constr::search_penalty");
    scalar_t sum = 0.;
    for_each_box_side(d, box, [&](auto, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &present) {
        const auto safe_slack = present.select(pair.slack.array(), scalar_t(1));
        sum += present.select(safe_slack.log(), scalar_t(0)).sum();
    });
    return d.ipm_cfg->mu * sum;
}
scalar_t ipm_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const ipm_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("ipm_constr::search_penalty_dir_deriv");
    scalar_t sum = 0.;
    for_each_box_side(d, box, [&](auto, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &present) {
        const auto safe_backup = present.select(pair.slack_backup.array(), scalar_t(1));
        sum += present.select(pair.d_slack.array() / safe_backup, scalar_t(0)).sum();
    });
    return d.ipm_cfg->mu * sum;
}

generic_constr::residual_summary ipm_constr::primal_residual_summary(const func_approx_data &data) const {
    const auto &d = static_cast<const ipm_data &>(data);
    const auto &box = d.require_box_spec("ipm_constr::primal_residual_summary");
    residual_summary summary{};
    for_each_box_side(d, box, [&](auto, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &present) {
        const auto lifted =
            present.select((pair.residual.array() + pair.slack.array()).abs(), scalar_t(0));
        summary.inf = std::max(summary.inf, lifted.size() > 0 ? lifted.maxCoeff() : scalar_t(0.));
        summary.l1 += lifted.sum();
    });
    return summary;
}

void ipm_constr::finalize_predictor_step(ipm::data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<ipm_data>();
    auto &ipm_worker = cfg->as<ipm_config::worker_type>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    const auto &box = d.require_box_spec("ipm_constr::finalize_predictor_step");
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &side_data, const box_mask &present) {
        ipm_worker.n_ipm_cstr += static_cast<size_t>(present.count());
        ipm_worker.prev_aff_comp +=
            present.select(pair.multiplier.array() * pair.slack.array(), scalar_t(0)).sum();
        side_data.corrector.array() =
            ls_cfg.alpha_dual * pair.d_multiplier.array() * ls_cfg.alpha_primal * pair.d_slack.array();
        ipm_worker.post_aff_comp +=
            present.select((pair.multiplier + ls_cfg.alpha_dual * pair.d_multiplier).array() *
                               (pair.slack + ls_cfg.alpha_primal * pair.d_slack).array(),
                           scalar_t(0))
                .sum();
    });
    // assert((upper_pair_data.multiplier.dot(upper_pair_data.slack) + lower_pair_data.multiplier.dot(lower_pair_data.slack)) > 0 &&
    //        "the complementarity must be positive before the line search step");
}
void ipm_constr::apply_affine_step(ipm::data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<ipm_data>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    const auto &box = d.require_box_spec("ipm_constr::apply_affine_step");
    assert(!d.ipm_cfg->ipm_computing_affine_step() && "ipm affine step computation not ended");
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &) {
        positivity::apply_pair_step(pair.slack, pair.d_slack, ls_cfg.alpha_primal,
                                    pair.multiplier, pair.d_multiplier, ls_cfg.dual_alpha_for_ineq());
    });
    refresh_box_state(d);
    if (d.ipm_cfg->ipm_accept_corrector()) {
        for_each_box_side(d, box, [](auto, box_pair_runtime &pair, ipm_side_data &, const box_mask &present) {
            pair.slack.array() = present.select(pair.slack.array().max(1e-20), scalar_t(0));
            pair.multiplier.array() = present.select(pair.multiplier.array().max(1e-20), scalar_t(0));
        });
        refresh_box_state(d);
    }
    // correction
    // constexpr scalar_t scale = 1e10;
    // each side would need its own clamping here if we ever re-enable this path
}
void ipm_constr::value_impl(func_approx_data &data) const {
    base::value_impl(data);
    auto &d = data.as<ipm_data>();
    solver::ineq_soft::ensure_initialized(*this, d);
    const auto &box = d.require_box_spec("ipm_constr::value_impl");
    for_each_box_side(d, box, [](auto, box_pair_runtime &, ipm_side_data &ipm_side, const box_mask &) {
        ipm_side.reg.setConstant(1e-8);
    });
    refresh_box_state(d);
    if (d.v_.hasNaN()) {
        fmt::print("g: {}\n", d.v_.transpose());
        for_each_box_side(d, box, [](auto side, const box_pair_runtime &pair, const ipm_side_data &, const box_mask &) {
            fmt::print("{} slack: {}\n",
                       side == box_side::ub ? "ub" : "lb",
                       pair.slack.transpose());
        });
        throw std::runtime_error("ipm_constr value_impl failed due to NaN");
    }
}
void ipm_constr::jacobian_impl(func_approx_data &data) const {
    base::jacobian_impl(data);
    auto &d = data.as<ipm_data>();
    if (d.ipm_cfg != nullptr && d.ipm_cfg->disable_corrections) {
        return;
    }
    const auto &box = d.require_box_spec("ipm_constr::jacobian_impl");
    for_each_box_side(d, box, [&](auto, box_pair_runtime &pair, ipm_side_data &ipm_side, const box_mask &present) {
        compute_side_reg_scaling(pair,
                                 ipm_side,
                                 present,
                                 d.ipm_cfg->mu,
                                 !d.ipm_cfg->ipm_enable_affine_step());
    });
    propagate_jacobian(d);
    propagate_hessian(d);
}

void ipm_constr::propagate_jacobian(func_approx_data &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::propagate_jacobian");
    for_each_box_side(d, box, [&](auto side, const box_pair_runtime &, const ipm_side_data &ipm_side, const box_mask &) {
        const scalar_t sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
        soft_constr::propagate_jacobian(d, ipm_side.scaled_res, sign);
    });
}

void ipm_constr::propagate_hessian(func_approx_data &data) const {
    auto &d = data.as<ipm_data>();
    const auto &box = d.require_box_spec("ipm_constr::propagate_hessian");
    for_each_box_side(d, box, [&](auto, const box_pair_runtime &, const ipm_side_data &ipm_side, const box_mask &) {
        soft_constr::propagate_hessian(d, ipm_side.diag_scaling);
    });
}
} // namespace solver
} // namespace moto
