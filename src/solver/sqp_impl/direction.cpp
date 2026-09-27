#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>
#include <moto/core/linear_backend.hpp>
#include <moto/utils/field_conversion.hpp>

#include <cmath>

namespace moto {
namespace {
scalar_t inf_norm(const auto &v) {
    return v.size() ? v.cwiseAbs().maxCoeff() : 0.;
}
scalar_t operator_norm(const sparse_matrix &a, bool transpose = false) {
    if (a.is_empty()) return 0.;
    const matrix dense = a.dense().cwiseAbs();
    return transpose ? dense.colwise().sum().maxCoeff() : dense.rowwise().sum().maxCoeff();
}
}

void ns_sqp::check_direction() {
    auto &graph = active_data();
    // The backward pass mutates gradients. Evaluate the full-space equations
    // from their saved stage gradients, not from the propagated value function.
    solver::for_each(solver::par, graph, [this](data *d) {
        riccati_solver_.finalize_dual_newton_step(d);
        riccati_solver_.compute_kkt_residual(d);
    });
    std::vector<array_type<scalar_t, primal_fields>> scales(graph.nodes().size());
    for (size_t k = 0; k < graph.nodes().size(); ++k) {
        auto &d = *graph.nodes()[k];
        auto &raw = d.dense();
        for (auto f : primal_fields) {
            if (!d.trial_prim_step[f].allFinite() || !d.kkt_stat_err_[f].allFinite()) {
                linear_solve_last.status = linear_solve_status::nonfinite_direction;
                return;
            }
            scalar_t scale = inf_norm(d.base_lag_grad_backup[f]);
            for (auto g : primal_fields) {
                const auto hi = std::max(f, g), lo = std::min(f, g);
                scale += (operator_norm(raw.lag_hess_[hi][lo], f < g) +
                          operator_norm(raw.constraint_hess_[hi][lo], f < g)) * inf_norm(d.trial_prim_step[g]);
            }
            scale += d.primal_regularization[f] * inf_norm(d.trial_prim_step[f]);
            for (auto c : constr_fields) {
                if (!d.trial_dual_step[c].allFinite()) {
                    linear_solve_last.status = linear_solve_status::nonfinite_direction;
                    return;
                }
                scale += operator_norm(raw.approx_[c].jac_[f], true) * inf_norm(d.trial_dual_step[c]);
            }
            if (!std::isfinite(scale)) {
                linear_solve_last.status = linear_solve_status::nonfinite_direction;
                return;
            }
            scales[k][f] = scale;
        }
        for (auto c : hard_constr_fields) {
            vector residual = raw.approx_[c].v_;
            scalar_t scale = 1. + inf_norm(residual);
            for (auto f : primal_fields) {
                vector product = vector::Zero(residual.size());
                linear_backend::multiply(raw.approx_[c].jac_[f], d.trial_prim_step[f], product);
                residual += product;
                scale += inf_norm(product);
            }
            if (!residual.allFinite() || !std::isfinite(scale)) {
                linear_solve_last.status = linear_solve_status::nonfinite_direction;
                return;
            }
            linear_solve_last.equality_residual = std::max(
                linear_solve_last.equality_residual, inf_norm(residual) / scale);
        }
        d.for_each<ineq_soft_constr_fields>([&](const soft_constr &, soft_constr::data_map_t &base) {
            using namespace solver::restoration;
            if (const auto *p = dynamic_cast<const solver::pmm_constr::approx_data *>(&base)) {
                const vector residual = p->g_ + p->jac_step_ - p->rho_ * p->d_multiplier_;
                if (!residual.allFinite()) {
                    linear_solve_last.status = linear_solve_status::nonfinite_direction;
                    return;
                }
                const scalar_t scale = 1. + inf_norm(p->g_) + inf_norm(p->jac_step_) + p->rho_ * inf_norm(p->d_multiplier_);
                linear_solve_last.inequality_residual = std::max(linear_solve_last.inequality_residual, inf_norm(residual) / scale);
            }
            const auto record_elastic = [&](const auto &local, const auto &delta, const auto &summary) {
                scalar_t scale = 1. + inf_norm(delta);
                bool finite = delta.allFinite();
                const auto add_side = [&](const auto &side) {
                    for (auto slot : {detail::slot_p, detail::slot_n}) {
                        finite &= side.r_stat[slot].allFinite();
                        scale += inf_norm(side.r_stat[slot]);
                    }
                    using slots = std::decay_t<decltype(side.value)>;
                    for (size_t slot = slots::shift_size::value; slot < slots::shift_size::value + side.value.size(); ++slot) {
                        finite &= side.value[slot].allFinite() && side.dual[slot].allFinite() &&
                                  side.d_value[slot].allFinite() && side.d_dual[slot].allFinite() && side.r_comp[slot].allFinite();
                        scale += inf_norm(side.d_value[slot]) + inf_norm(side.d_dual[slot]) + inf_norm(side.r_comp[slot]) +
                                 inf_norm(side.dual[slot]) * inf_norm(side.d_value[slot]) +
                                 inf_norm(side.value[slot]) * inf_norm(side.d_dual[slot]);
                    }
                };
                if constexpr (requires { local.side; }) {
                    for (auto side : box_sides) if (local.present_mask[side].any()) {
                        finite &= local.side[side].r_d.allFinite();
                        scale += inf_norm(local.side[side].r_d);
                        add_side(local.side[side]);
                    }
                } else {
                    finite &= local.r_c.allFinite() && local.d_multiplier.allFinite();
                    scale += inf_norm(local.r_c) + inf_norm(local.d_multiplier);
                    add_side(local);
                }
                if (!finite || !std::isfinite(scale)) {
                    linear_solve_last.status = linear_solve_status::nonfinite_direction;
                    return;
                }
                linear_solve_last.inequality_residual = std::max({linear_solve_last.inequality_residual,
                    summary.inf_prim / scale, summary.inf_stat / scale, summary.inf_comp / scale});
            };
            if (const auto *p = dynamic_cast<const resto_eq_elastic_constr::approx_data *>(&base))
                record_elastic(p->elastic, p->jac_step, resto_eq_elastic_constr::linearized_newton_residuals(p->jac_step, p->elastic));
            if (const auto *p = dynamic_cast<const resto_ineq_elastic_ipm_constr::approx_data *>(&base))
                record_elastic(p->elastic, p->jac_step, resto_ineq_elastic_ipm_constr::linearized_newton_residuals(p->jac_step, p->elastic));
        });
        d.for_each<ineq_constr_fields>([&](const ineq_constr &, ineq_constr::data_map_t &base) {
            const auto *ipm = dynamic_cast<const solver::ipm_constr::ipm_data *>(&base);
            if (!ipm) return; // elastic equations were checked above
            const auto &box = ipm->require_box_spec("check_direction");
            for (auto side : box_sides) if (box.has_side[side]) {
                const auto &p = static_cast<const solver::ipm_constr::approx_data::side_data &>(*ipm->box_side_[side]);
                for (Eigen::Index i = 0; i < p.slack.size(); ++i) if (box.present_mask[side](i)) {
                    const scalar_t jdx = (side == box_side::ub ? 1. : -1.) * ipm->jac_step(i);
                    const scalar_t primal = p.residual(i) + p.slack(i) + jdx + p.d_slack(i) - p.reg(i) * p.d_multiplier(i);
                    const scalar_t comp = p.slack(i) * p.multiplier(i) + p.multiplier(i) * p.d_slack(i) + p.slack(i) * p.d_multiplier(i) - settings.ipm.mu + (settings.ipm.ipm_accept_corrector() ? p.corrector(i) : 0.);
                    const scalar_t primal_scale = 1. + std::abs(p.residual(i)) + p.slack(i) + std::abs(jdx) + std::abs(p.d_slack(i)) + std::abs(p.reg(i) * p.d_multiplier(i));
                    const scalar_t comp_scale = 1. + p.slack(i) * p.multiplier(i) + std::abs(p.multiplier(i) * p.d_slack(i)) + std::abs(p.slack(i) * p.d_multiplier(i)) + settings.ipm.mu;
                    if (!std::isfinite(primal) || !std::isfinite(comp)) {
                        linear_solve_last.status = linear_solve_status::nonfinite_direction;
                        return;
                    }
                    linear_solve_last.inequality_residual = std::max({linear_solve_last.inequality_residual,
                        std::abs(primal) / primal_scale, std::abs(comp) / comp_scale});
                }
            }
        });
    }
    for (size_t k = 0; k < graph.nodes().size(); ++k) {
        auto &d = *graph.nodes()[k];
        // x[0] is fixed; every other x is the previous interval's y. The
        // optimized initial state is represented by an existing virtual stage.
        row_vector state_residual = d.kkt_stat_err_[__y];
        scalar_t state_scale = scales[k][__y];
        if (k + 1 < graph.nodes().size()) {
            auto &next = *graph.nodes()[k + 1];
            state_residual += (next.kkt_stat_err_[__x] *
                utils::permutation_from_y_to_x(&d.problem(), &next.problem())).eval();
            state_scale += scales[k + 1][__x];
        }
        linear_solve_last.stationarity_residual = std::max({linear_solve_last.stationarity_residual,
            inf_norm(state_residual) / (1. + state_scale),
            inf_norm(d.kkt_stat_err_[__u]) / (1. + scales[k][__u]),
            inf_norm(d.kkt_stat_err_[__l]) / (1. + scales[k][__l])});
    }
    const auto tol = settings.regularization.residual_tolerance;
    if (linear_solve_last.status == linear_solve_status::nonfinite_direction) return;
    if (linear_solve_last.equality_residual > tol)
        linear_solve_last.status = linear_solve_status::inconsistent_equalities;
    else if (std::max(linear_solve_last.stationarity_residual, linear_solve_last.inequality_residual) > tol)
        linear_solve_last.status = linear_solve_status::inaccurate_direction;
}

bool ns_sqp::compute_safe_direction(iteration_context &ctx, bool do_scaling,
                                    bool do_refinement, bool gauss_newton) {
    const auto &cfg = settings.regularization;
    if (!std::isfinite(cfg.initial) || cfg.initial <= 0. ||
        !std::isfinite(cfg.maximum) || cfg.maximum < cfg.initial ||
        !std::isfinite(cfg.increase_factor) || cfg.increase_factor <= 1. ||
        !std::isfinite(cfg.decrease_factor) || cfg.decrease_factor <= 0. || cfg.decrease_factor >= 1. ||
        !std::isfinite(cfg.residual_tolerance) || cfg.residual_tolerance <= 0. || cfg.max_attempts == 0)
        throw std::invalid_argument("invalid adaptive regularization settings");
    auto &graph = active_data();
    const solver::ipm_config ipm_before = settings.ipm;
    for (auto *d : graph.nodes()) d->backup_trial_state();
    scalar_t delta = 0.;
    const size_t attempts = cfg.enabled ? cfg.max_attempts : 1;
    for (size_t attempt = 0; attempt < attempts; ++attempt) {
        if (attempt) {
            static_cast<solver::ipm_config &>(settings.ipm) = ipm_before;
            ctx.mu_changed = false;
            solver::for_each(solver::par, graph, [](data *d) {
                d->restore_trial_state();
                d->update_approximation(node_data::update_mode::eval_all);
            });
        }
        reset_ls_workers();
        linear_solve_last = {};
        linear_solve_last.attempts = attempt + 1;
        linear_solve_last.regularization = delta;
        for (auto *d : graph.nodes()) {
            for (auto f : primal_fields) {
                // State ownership is y, never both x and the preceding y.
                // The virtual initial input is another copy of its y state.
                const scalar_t amount = f == __x || (d->internal_initial_state && f == __u) ? 0. : delta;
                d->primal_regularization[f] = amount;
                auto &h = d->dense().hessian_modification_[f][f];
                if (amount && h.rows())
                    h.view(0, 0, h.rows(), h.cols(), sparsity::diag).array() += amount;
            }
        }
        try {
            solve_direction(ctx, do_scaling, gauss_newton);
            correct_direction(ctx, do_refinement);
            check_direction();
        } catch (const solver::ns_riccati::factorization_failure &error) {
            linear_solve_last.status = linear_solve_status::factorization_failed;
            if (settings.verbose) fmt::println("[linear solve] {}", error.what());
        }
        if (linear_solve_last.status == linear_solve_status::success) {
            last_primal_regularization_ = delta;
            return true;
        }
        if (settings.verbose)
            fmt::println("[linear solve] attempt={} delta={} status={} residuals=({},{},{})", attempt + 1, delta,
                static_cast<int>(linear_solve_last.status), linear_solve_last.stationarity_residual,
                linear_solve_last.equality_residual, linear_solve_last.inequality_residual);
        // Primal damping cannot repair a dropped/inconsistent hard equation.
        if (linear_solve_last.status == linear_solve_status::inconsistent_equalities) break;
        delta = delta == 0. ? std::max(cfg.initial, last_primal_regularization_ * cfg.decrease_factor)
                           : delta * cfg.increase_factor;
        if (!std::isfinite(delta) || delta > cfg.maximum) break;
    }
    static_cast<solver::ipm_config &>(settings.ipm) = ipm_before;
    ctx.mu_changed = false;
    solver::for_each(solver::par, graph, [](data *d) {
        d->restore_trial_state();
        d->update_approximation(node_data::update_mode::eval_all);
        for (auto f : primal_fields) { d->trial_prim_step[f].setZero(); d->primal_regularization[f] = 0.; }
        for (auto f : constr_fields) d->trial_dual_step[f].setZero();
    });
    return false;
}
} // namespace moto
