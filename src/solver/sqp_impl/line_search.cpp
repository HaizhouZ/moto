#include <algorithm>
#include <moto/solver/ns_sqp.hpp>
// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#define SHOW_DETAIL_TIMING
#include <moto/solver/ineq_soft.hpp>

namespace moto {

void ns_sqp::finalize_ls_bound_and_set_to_max() {
    // merge line search bounds from each thread
    for (solver::linesearch_config &s : setting_per_thread) {
        settings.ls.primal.merge_from(s.primal);
        settings.ls.dual.merge_from(s.dual);
    }
    settings.ls.alpha_primal = settings.ls.primal.alpha_max;
    settings.ls.alpha_dual = settings.ls.dual.alpha_max;
    // copy the settings to each worker
    for (solver::linesearch_config &s : setting_per_thread) {
        s.copy_from(settings.ls);
    }
}

void ns_sqp::filter_linesearch_data::update_filter(const kkt_info &current_kkt, settings_t &settings) {
    // only add to the filter if the new point is not dominated by the filter
    point current_point{
        .prim_res = current_kkt.prim_res_l1,
        .dual_res = current_kkt.inf_dual_res,
        .objective = current_kkt.objective,
    };
    points.erase(std::remove_if(
                     points.begin(), points.end(),
                     [&](const point &p) {
                         return p.in_filter(current_point, settings);
                     }),
                 points.end());
    if (settings.verbose)
        fmt::print("  added point to filter: primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n",
                   current_point.prim_res, current_point.dual_res, current_point.objective);
    points.push_back(current_point);
}
bool ns_sqp::filter_linesearch_data::point::in_filter(const point &filter_entry, const settings_t &settings) const {
    return prim_res >= (1.0 - settings.ls.primal_gamma) * filter_entry.prim_res and
           objective >= filter_entry.objective - settings.ls.dual_gamma * filter_entry.prim_res;
}
bool ns_sqp::filter_linesearch_data::try_step(const kkt_info &trial_kkt, const kkt_info &current_kkt, settings_t &settings) {
    scalar_t prim_res_k = current_kkt.prim_res_l1;
    scalar_t obj_trial = trial_kkt.objective;
    // recompute with current mu in case mu changed after current_kkt was computed
    scalar_t mu = settings.ipm.mu;
    scalar_t obj_k = current_kkt.cost - mu * current_kkt.log_slack_sum;
    scalar_t fullstep_dec_k = current_kkt.obj_fullstep_dec - mu * current_kkt.barrier_dir_deriv;
    point trial_point{
        .prim_res = trial_kkt.prim_res_l1,
        .dual_res = trial_kkt.inf_dual_res,
        .objective = trial_kkt.objective,
    };

    // switching condition based on IPOPT's filter line search paper: https://www.coin-or.org/Ipopt/documentation/node40.html#SECTION00421000000000000000
    switching_condition =
        fullstep_dec_k < 0.0 &&
        settings.ls.alpha_primal * std::pow(-fullstep_dec_k, settings.ls.s_phi) >= std::pow(current_kkt.prim_res_l1, settings.ls.s_theta);

    // Armijo condition for the objective
    scalar_t armijo_target = obj_k +
                             settings.ls.armijo_dec_frac * settings.ls.alpha_primal * fullstep_dec_k;
    armijo_cond_met = obj_trial <= armijo_target;

    if (settings.verbose) {
        fmt::print("  switching condition: {}, armijo condition: {}\n", switching_condition ? "met" : "not met", armijo_cond_met ? "met" : "not met");
        fmt::print("  cost step dec: {:.3e}, full step decrease: {:.3e}, switching lhs: {:.3e}, switching rhs: {:.3e}\n",
                   current_kkt.obj_fullstep_dec, fullstep_dec_k, settings.ls.alpha_primal * std::pow(-fullstep_dec_k, settings.ls.s_phi), std::pow(current_kkt.prim_res_l1, settings.ls.s_theta));
    }

    // Flat-objective accept: when the directional derivative is negligibly small, the iterate is
    // nearly feasible, and the step is non-trivial, accept to allow dual progress without requiring
    // objective decrease (the objective is flat so the filter/Armijo conditions would stall).
    auto check_flat_obj = [&]() {
        if (settings.ls.enable_flat_obj_accept)
            if (std::abs(fullstep_dec_k) <= settings.ls.flat_obj_dec_tol * (1 + std::abs(obj_k)) &&
                prim_res_k < settings.ls.flat_obj_prim_tol &&
                current_kkt.inf_comp_res < settings.comp_tol &&
                current_kkt.inf_dual_step > std::max(settings.ls.flat_obj_step_tol, settings.dual_tol)) {
                last_step_was_armijo = false;
                if (settings.verbose)
                    fmt::print("  trial point accepted by flat-objective condition (fullstep_dec={:.3e}, prim_res={:.3e}, comp_res={:.3e}, step_norm={:.3e})\n",
                               fullstep_dec_k, prim_res_k, current_kkt.inf_comp_res, current_kkt.inf_dual_step);
                return true;
            } else {
                if (settings.verbose)
                    fmt::print("  flat-objective accept condition not met (fullstep_dec={:.3e}, prim_res={:.3e}, comp_res={:.3e}, step_norm={:.3e})\n",
                               fullstep_dec_k, prim_res_k, current_kkt.inf_comp_res, current_kkt.inf_dual_step);
                return false;
            }
        return false;
    };

    // reject if the trial point is dominated by any point in the filter
    for (const auto &p : points) {
        if (trial_point.in_filter(p, settings)) {
            filter_reject_cnt++;
            if (settings.verbose) {
                fmt::print("  trial point rejected by filter (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}), dominated by filter point (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e})\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective,
                           p.prim_res, p.dual_res, p.objective);
            }
            return check_flat_obj();
        }
    }
    filter_reject_cnt = 0;

    // sufficient objective decrease condition (Armijo) for switching
    if (switching_condition && prim_res_k <= constr_vio_min) {
        if (armijo_cond_met) {
            last_step_was_armijo = true;
            if (settings.verbose) {
                fmt::print("  trial point accepted by Armijo condition in switching mode (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}), armijo target: {:.3e}\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective, armijo_target);
            }
            return true;
        } else {
            if (settings.verbose) {
                fmt::print("  trial point rejected by Armijo condition in switching mode (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}), armijo target: {:.3e}\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective, armijo_target);
            }
            return check_flat_obj();
        }
    } else { // filter condition for non-switching
        // Filter condition relative to the CURRENT iterate
        // pass if the trial point makes sufficient progress in either primal residual or objective compared to the current point (not the filter points)
        point current_point{
            .prim_res = current_kkt.prim_res_l1,
            .dual_res = current_kkt.inf_dual_res,
            .objective = current_kkt.objective,
        };
        if (!trial_point.in_filter(current_point, settings)) {
            last_step_was_armijo = false;
            if (settings.verbose)
                fmt::print("  trial point accepted by filter condition in non-switching mode (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}) sufficient progress wrt current point (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e})\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective,
                           current_point.prim_res, current_point.dual_res, current_point.objective);
            return true;
        } else {
            if (settings.verbose) {
                fmt::print("  trial point rejected by filter condition in non-switching mode (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}), dominated by current point (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e})\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective,
                           current_point.prim_res, current_point.dual_res, current_point.objective);
            }
            return check_flat_obj();
        }
    }
}

bool ns_sqp::outer_filter_accepts(const filter_linesearch_data &ls,
                                  const kkt_info &trial_kkt,
                                  const kkt_info &reference_kkt) {
    const scalar_t mu = settings.ipm.mu;
    const scalar_t obj_trial = trial_kkt.objective;
    const scalar_t obj_k = reference_kkt.cost - mu * reference_kkt.log_slack_sum;
    const scalar_t fullstep_dec_k = reference_kkt.obj_fullstep_dec - mu * reference_kkt.barrier_dir_deriv;
    const filter_linesearch_data::point trial_point{
        .prim_res = trial_kkt.prim_res_l1,
        .dual_res = trial_kkt.inf_dual_res,
        .objective = trial_kkt.objective,
    };
    const filter_linesearch_data::point reference_point{
        .prim_res = reference_kkt.prim_res_l1,
        .dual_res = reference_kkt.inf_dual_res,
        .objective = reference_kkt.objective,
    };

    for (const auto &p : ls.points) {
        if (trial_point.in_filter(p, settings)) {
            return false;
        }
    }

    const bool switching_condition =
        fullstep_dec_k < 0.0 &&
        settings.ls.alpha_primal * std::pow(-fullstep_dec_k, settings.ls.s_phi) >= std::pow(reference_kkt.prim_res_l1, settings.ls.s_theta);
    const scalar_t armijo_target =
        obj_k + settings.ls.armijo_dec_frac * settings.ls.alpha_primal * fullstep_dec_k;
    const bool armijo_cond_met = obj_trial <= armijo_target;

    if (switching_condition && reference_kkt.prim_res_l1 <= ls.constr_vio_min) {
        return armijo_cond_met;
    }
    return !trial_point.in_filter(reference_point, settings);
}
void ns_sqp::step_back_alpha(filter_linesearch_per_iter_data &ls) {
    if (settings.ls.backtrack_scheme == linesearch_setting::backtrack_scheme_t::geometric)
        settings.ls.alpha_primal *= settings.ls.backtrack_factor;
    else
        settings.ls.alpha_primal = std::max(
            settings.ls.alpha_primal - ls.initial_alpha_primal / (settings.ls.max_steps + 1e-8),
            scalar_t(0.0));
    if (settings.ls.update_alpha_dual) {
        if (settings.ls.backtrack_scheme == linesearch_setting::backtrack_scheme_t::geometric)
            settings.ls.alpha_dual *= settings.ls.backtrack_factor;
        else
            settings.ls.alpha_dual = std::max(
                settings.ls.alpha_dual - ls.initial_alpha_dual / (settings.ls.max_steps + 1e-8),
                scalar_t(0.0));
    }
}

ns_sqp::line_search_action ns_sqp::filter_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt) {
    if (ls.step_cnt == 0 && trial_kkt.prim_res_l1 < current_kkt.prim_res_l1) {
        // skip second-order correction if the first trial already shows improvement in primal residual
        ls.skip_soc = true;
    }
    scalar_t mu = settings.ipm.mu;
    scalar_t fullstep_dec_k = current_kkt.obj_fullstep_dec - mu * current_kkt.barrier_dir_deriv;
    ls.alpha_min = settings.ls.primal.alpha_min;

    const auto record_best_trial = [&] {
        const auto make_point = [&](const kkt_info &kkt) -> filter_linesearch_data::point {
            return {
                .prim_res = kkt.prim_res_l1,
                .dual_res = kkt.inf_dual_res,
                .objective = kkt.objective,
            };
        };
        auto trial_point = make_point(trial_kkt);
        bool prim_better = trial_point.prim_res < ls.best_trial.prim_res;
        bool obj_better = trial_point.objective < ls.best_trial.objective;
        if (prim_better || obj_better) {
            ls.best_trial.prim_res = trial_point.prim_res;
            ls.best_trial.dual_res = trial_point.dual_res;
            ls.best_trial.objective = trial_point.objective;
            ls.best_trial.alpha_primal = settings.ls.alpha_primal;
            ls.best_trial.alpha_dual = settings.ls.alpha_dual;
        }
    };
    const auto print_filter = [&] {
        if (!settings.verbose)
            return;
        for (size_t i = 0; i < ls.points.size(); ++i) {
            fmt::print("   filter point {}: primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n", i, ls.points[i].prim_res, ls.points[i].dual_res, ls.points[i].objective);
        }
    };

    record_best_trial();
    if (settings.verbose) {
        fmt::print("[ls] step no. {}, primal res: {:.3e}, barrier obj: {:.3e}, alpha_primal: {:.3e}, alpha_dual: {:.3e}\n",
                   ls.step_cnt, trial_kkt.prim_res_l1, trial_kkt.objective, settings.ls.alpha_primal, settings.ls.alpha_dual);
        fmt::print("  alpha_min: {:.3e}\n", ls.alpha_min);
    }
    print_filter();
    bool accept = ls.try_step(trial_kkt, current_kkt, settings);
    /// if the point is acceptable or we have already tried enough steps, stop line search and accept the point if acceptable
    if (accept || ls.stop) {
        ls.recompute_approx = false;
        if (accept && !ls.stop) {
            if (!ls.switching_condition || !ls.armijo_cond_met) {
                // only update the filter if we are not in switching mode with Armijo condition met,
                // otherwise the filter is not relevant and may even reject future points that make sufficient primal progress
                // but not sufficient objective decrease
                ls.update_filter(current_kkt, settings);
            }
        }
        return accept ? line_search_action::accept : line_search_action::failure;
    }

    ls.recompute_approx = true;
    // try second-order correction (IPOPT §3.4 / A-5.5): only on the first rejection,
    // only when rejected by filter (not Armijo), and κ_soc abort not triggered.
    if (settings.ls.enable_soc && ls.step_cnt == 0 && !ls.skip_soc &&
        ls.soc_iter_cnt < settings.ls.max_soc_iter && !ls.switching_condition) {
    }

    if (settings.ls.alpha_primal <= ls.alpha_min &&
        current_kkt.prim_res_l1 > settings.prim_tol) {
        ls.stop = true;
        ls.failure_reason = filter_linesearch_per_iter_data::failure_reason_t::tiny_step;
        ls.recompute_approx = false;
        if (settings.verbose) {
            fmt::print("  line search reached min step: alpha_p {:.3e} <= alpha_min {:.3e} with prim_res {:.3e}\n",
                       settings.ls.alpha_primal, ls.alpha_min, current_kkt.prim_res_l1);
        }
        return line_search_action::failure;
    }

    /// backtrack
    if (settings.ls.max_steps > ls.step_cnt) {
        ls.step_cnt++;
        step_back_alpha(ls);
        if (settings.verbose)
            fmt::print("  backtrack, alpha_p: {:.3e}, alpha_d: {:.3e}\n", settings.ls.alpha_primal, settings.ls.alpha_dual);
        return line_search_action::backtrack;
    }
    // line search failed, fallback to backup
    ls.stop = true; // stop line search, will not try more steps
    ls.failure_reason = filter_linesearch_per_iter_data::failure_reason_t::other;
    if (settings.ls.failure_strategy == linesearch_setting::failure_backup_strategy::min_step) {
        if (settings.verbose) {
            fmt::print("  ls failed, use min primal step...\n");
        }
        ls.enforce_min = true;
        auto scale = std::min(0.01 / ls.initial_alpha_primal, 1.0);
        settings.ls.alpha_primal = ls.initial_alpha_primal * scale;
    } else {
        if (settings.verbose) {
            fmt::print("  ls failed, use best trial...\n");
            fmt::print("    best trial primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}, alpha_p: {:.3e}, alpha_d: {:.3e}\n",
                       ls.best_trial.prim_res, ls.best_trial.dual_res, ls.best_trial.objective, ls.best_trial.alpha_primal, ls.best_trial.alpha_dual);
        }
        settings.ls.alpha_primal = ls.best_trial.alpha_primal;
        if (settings.ls.update_alpha_dual) {
            settings.ls.alpha_dual = ls.best_trial.alpha_dual;
        }
    }

    fmt::println(" line search failed, dec_full_pred = {:.3e}, best trial primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n",
                 fullstep_dec_k, ls.best_trial.prim_res, ls.best_trial.dual_res, ls.best_trial.objective);
    ls.update_filter(current_kkt, settings);
    return line_search_action::failure;
}

ns_sqp::line_search_action ns_sqp::merit_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt) {
    const auto merit = [&](scalar_t prim_l1, scalar_t dual_res) -> scalar_t {
        return prim_l1 * prim_l1 + settings.ls.merit_sigma * dual_res * dual_res;
    };

    scalar_t merit_trial = merit(trial_kkt.prim_res_l1, trial_kkt.avg_dual_res);
    scalar_t merit_k = merit(current_kkt.prim_res_l1, current_kkt.avg_dual_res);

    // On the first (full-step) trial, record merit to estimate the directional derivative.
    // dir_deriv ≈ (M(x + 1*d) - M(x)) / 1.0  (finite-difference estimate)
    if (ls.step_cnt == 0) {
        ls.merit_fullstep = merit_trial;
    }

    // Track best trial for the fallback strategy
    if (merit_trial < ls.best_merit_trial.merit) {
        ls.best_merit_trial.merit = merit_trial;
        ls.best_merit_trial.alpha_primal = settings.ls.alpha_primal;
        ls.best_merit_trial.alpha_dual = settings.ls.alpha_dual;
    }

    if (settings.verbose) {
        fmt::print("[merit ls] step {}, merit: {:.3e} (prim: {:.3e}, avg_dual: {:.3e}), alpha_p: {:.3e}, merit_k: {:.3e} (prim: {:.3e}, avg_dual: {:.3e})\n",
                   ls.step_cnt, merit_trial, trial_kkt.prim_res_l1, trial_kkt.avg_dual_res,
                   settings.ls.alpha_primal, merit_k, current_kkt.prim_res_l1, current_kkt.avg_dual_res);
    }

    // Armijo sufficient decrease: M(x + alpha*d) <= M(x) + c * alpha * dir_deriv
    // dir_deriv estimated from the full step (negative when making progress).
    scalar_t dir_deriv = ls.merit_fullstep - merit_k;
    scalar_t armijo_target = merit_k + settings.ls.armijo_dec_frac * settings.ls.alpha_primal * dir_deriv;

    bool accept = merit_trial <= armijo_target;

    if (accept || ls.stop) {
        ls.recompute_approx = false;
        return accept ? line_search_action::accept : line_search_action::failure;
    }

    ls.recompute_approx = true;
    if (settings.ls.max_steps > ls.step_cnt) {
        ls.step_cnt++;
        step_back_alpha(ls);
        if (settings.verbose)
            fmt::print("  merit backtrack, alpha_p: {:.3e}\n", settings.ls.alpha_primal);
        return line_search_action::backtrack;
    }

    // Line search failed — apply fallback
    ls.stop = true;
    if (settings.ls.failure_strategy == linesearch_setting::failure_backup_strategy::min_step) {
        if (settings.verbose)
            fmt::print("  merit ls failed, use min primal step...\n");
        ls.enforce_min = true;
        auto scale = std::min(0.01 / ls.initial_alpha_primal, 1.0);
        settings.ls.alpha_primal = ls.initial_alpha_primal * scale;
    } else {
        if (settings.verbose) {
            fmt::print("  merit ls failed, use best trial (merit: {:.3e}, alpha_p: {:.3e})...\n",
                       ls.best_merit_trial.merit, ls.best_merit_trial.alpha_primal);
        }
        settings.ls.alpha_primal = ls.best_merit_trial.alpha_primal;
        if (settings.ls.update_alpha_dual)
            settings.ls.alpha_dual = ls.best_merit_trial.alpha_dual;
    }
    fmt::println(" merit line search failed, merit_k: {:.3e}, best merit: {:.3e}\n",
                 merit_k, ls.best_merit_trial.merit);
    return line_search_action::failure;
}

void ns_sqp::second_order_correction() {
}

} // namespace moto
