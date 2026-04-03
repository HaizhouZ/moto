#include <algorithm>
#include <cmath>
#include <moto/solver/ns_sqp.hpp>
// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#define SHOW_DETAIL_TIMING
#include <moto/solver/ineq_soft.hpp>

namespace moto {

scalar_t ns_sqp::current_linesearch_alpha_min(const filter_linesearch_per_iter_data &ls) const {
    scalar_t alpha_min = settings.ls.primal.alpha_min;
    if (settings.restoration.enabled)
        alpha_min = std::max(alpha_min,
                             settings.restoration.alpha_min_factor *
                                 std::max(ls.initial_alpha_primal, scalar_t(1e-12)));
    return alpha_min;
}

void ns_sqp::finalize_ls_bound_and_set_to_max() {
    for (solver::linesearch_config &s : setting_per_thread) {
        settings.ls.primal.merge_from(s.primal);
        settings.ls.dual.merge_from(s.dual);
    }
    settings.ls.alpha_primal = settings.ls.primal.alpha_max;
    settings.ls.alpha_dual   = settings.ls.dual.alpha_max;
    for (solver::linesearch_config &s : setting_per_thread)
        s.copy_from(settings.ls);
}

void ns_sqp::filter_linesearch_data::update_filter(const kkt_info &current_kkt, settings_t &settings) {
    const point current_point{
        .prim_res  = current_kkt.prim_res_l1,
        .dual_res  = current_kkt.inf_dual_res,
        .objective = current_kkt.penalized_obj,
    };
    points.erase(
        std::remove_if(points.begin(), points.end(),
                       [&](const point &p) { return p.in_filter(current_point, settings); }),
        points.end());
    if (settings.verbose)
        fmt::print("  added point to filter: primal res: {:.3e}, objective: {:.3e}\n",
                   current_point.prim_res, current_point.objective);
    points.push_back(current_point);
}

bool ns_sqp::filter_linesearch_data::point::in_filter(const point &e, const settings_t &s) const {
    return prim_res  >= (1.0 - s.ls.primal_gamma) * e.prim_res and
           objective >= e.objective - s.ls.dual_gamma * e.prim_res;
}

ns_sqp::filter_linesearch_data::normal_filter_eval_result
ns_sqp::filter_linesearch_data::evaluate_normal_filter_step(const std::vector<point> &filter_points,
                                                            const kkt_info &trial_kkt,
                                                            const kkt_info &current_kkt,
                                                            scalar_t constr_vio_min,
                                                            const settings_t &settings,
                                                            bool allow_flat_objective) {
    normal_filter_eval_result result{
        .trial_point =
            {
                .prim_res = trial_kkt.prim_res_l1,
                .dual_res = trial_kkt.inf_dual_res,
                .objective = trial_kkt.penalized_obj,
            },
        .current_point =
            {
                .prim_res = current_kkt.prim_res_l1,
                .dual_res = current_kkt.inf_dual_res,
                .objective = current_kkt.penalized_obj,
            },
    };
    const scalar_t prim_res_k = current_kkt.prim_res_l1;
    const scalar_t obj_k = current_kkt.penalized_obj;
    result.fullstep_dec = current_kkt.penalized_obj_fullstep_dec;
    result.switching_lhs =
        result.fullstep_dec < 0.0
            ? settings.ls.alpha_primal * std::pow(-result.fullstep_dec, settings.ls.s_phi)
            : scalar_t(0.0);
    result.switching_rhs = std::pow(prim_res_k, settings.ls.s_theta);
    result.switching_condition =
        result.fullstep_dec < 0.0 && result.switching_lhs >= result.switching_rhs;
    result.armijo_target =
        obj_k + settings.ls.armijo_dec_frac * settings.ls.alpha_primal * result.fullstep_dec;
    result.armijo_cond_met = trial_kkt.penalized_obj <= result.armijo_target;
    result.flat_objective_eligible =
        allow_flat_objective &&
        settings.ls.enable_flat_obj_accept &&
        std::abs(result.fullstep_dec) <=
            settings.ls.flat_obj_dec_tol * (1 + std::abs(obj_k)) &&
        prim_res_k < settings.ls.flat_obj_prim_tol &&
        current_kkt.inf_comp_res < settings.comp_tol &&
        current_kkt.inf_dual_step >
            std::max(settings.ls.flat_obj_step_tol, settings.dual_tol);

    for (const auto &p : filter_points) {
        if (result.trial_point.in_filter(p, settings)) {
            result.dominated_by_filter = true;
            result.dominating_filter_point = p;
            result.accepted_by_flat_objective = result.flat_objective_eligible;
            result.accepted = result.accepted_by_flat_objective;
            return result;
        }
    }

    if (result.switching_condition && prim_res_k <= constr_vio_min) {
        result.accepted_by_armijo = result.armijo_cond_met;
        result.accepted_by_flat_objective =
            !result.accepted_by_armijo && result.flat_objective_eligible;
        result.accepted = result.accepted_by_armijo || result.accepted_by_flat_objective;
        return result;
    }

    result.accepted_by_filter =
        !result.trial_point.in_filter(result.current_point, settings);
    result.accepted_by_flat_objective =
        !result.accepted_by_filter && result.flat_objective_eligible;
    result.accepted = result.accepted_by_filter || result.accepted_by_flat_objective;
    return result;
}

bool ns_sqp::filter_linesearch_data::try_step(const kkt_info &trial_kkt,
                                               const kkt_info &current_kkt,
                                               settings_t &settings) {
    auto log = [&]<typename... Args>(fmt::format_string<Args...> fmt_str, Args&&... args) {
        if (settings.verbose) fmt::print(fmt_str, std::forward<Args>(args)...);
    };

    const auto eval =
        evaluate_normal_filter_step(points, trial_kkt, current_kkt, constr_vio_min, settings);
    switching_condition = eval.switching_condition;
    armijo_cond_met = eval.armijo_cond_met;

    log("  switching condition: {}, armijo condition: {}\n",
        switching_condition ? "met" : "not met", armijo_cond_met ? "met" : "not met");
    log("  cost step dec: {:.3e}, full step decrease: {:.3e}, switching lhs: {:.3e}, switching rhs: {:.3e}\n",
        current_kkt.obj_fullstep_dec, eval.fullstep_dec, eval.switching_lhs, eval.switching_rhs);

    auto log_flat_obj = [&]() -> bool {
        if (eval.accepted_by_flat_objective) {
            last_step_was_armijo = false;
            log("  trial point accepted by flat-objective condition (fullstep_dec={:.3e}, prim_res={:.3e}, comp_res={:.3e}, step_norm={:.3e})\n",
                eval.fullstep_dec, current_kkt.prim_res_l1, current_kkt.inf_comp_res,
                current_kkt.inf_dual_step);
        } else {
            log("  flat-objective accept condition not met (fullstep_dec={:.3e}, prim_res={:.3e}, comp_res={:.3e}, step_norm={:.3e})\n",
                eval.fullstep_dec, current_kkt.prim_res_l1, current_kkt.inf_comp_res,
                current_kkt.inf_dual_step);
        }
        return eval.accepted_by_flat_objective;
    };

    if (eval.dominated_by_filter) {
        filter_reject_cnt++;
        log("  trial point rejected by filter (primal res: {:.3e}, objective: {:.3e}), dominated by filter point (primal res: {:.3e}, objective: {:.3e})\n",
            eval.trial_point.prim_res, eval.trial_point.objective,
            eval.dominating_filter_point.prim_res,
            eval.dominating_filter_point.objective);
        return log_flat_obj();
    }
    filter_reject_cnt = 0;

    if (switching_condition && current_kkt.prim_res_l1 <= constr_vio_min) {
        if (eval.accepted_by_armijo) {
            last_step_was_armijo = true;
            log("  trial point accepted by Armijo condition in switching mode (primal res: {:.3e}, objective: {:.3e}), armijo target: {:.3e}\n",
                eval.trial_point.prim_res, eval.trial_point.objective,
                eval.armijo_target);
            return true;
        }
        log("  trial point rejected by Armijo condition in switching mode (primal res: {:.3e}, objective: {:.3e}), armijo target: {:.3e}\n",
            eval.trial_point.prim_res, eval.trial_point.objective,
            eval.armijo_target);
        return log_flat_obj();
    }

    if (eval.accepted_by_filter) {
        last_step_was_armijo = false;
        log("  trial point accepted by filter condition in non-switching mode (primal res: {:.3e}, objective: {:.3e}) sufficient progress wrt current point (primal res: {:.3e}, objective: {:.3e})\n",
            eval.trial_point.prim_res, eval.trial_point.objective,
            eval.current_point.prim_res, eval.current_point.objective);
        return true;
    }
    log("  trial point rejected by filter condition in non-switching mode (primal res: {:.3e}, objective: {:.3e}), dominated by current point (primal res: {:.3e}, objective: {:.3e})\n",
        eval.trial_point.prim_res, eval.trial_point.objective,
        eval.current_point.prim_res, eval.current_point.objective);
    return log_flat_obj();
}

bool ns_sqp::outer_filter_accepts(const filter_linesearch_data &ls,
                                  const kkt_info &trial_kkt,
                                  const kkt_info &reference_kkt) {
    return filter_linesearch_data::evaluate_normal_filter_step(ls.points, trial_kkt, reference_kkt,
                                                               ls.constr_vio_min, settings, false)
        .accepted;
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

ns_sqp::line_search_action ns_sqp::filter_linesearch(filter_linesearch_data &ls,
                                                      const kkt_info &trial_kkt,
                                                      const kkt_info &current_kkt) {
    const filter_linesearch_data::point trial_point{
        .prim_res = trial_kkt.prim_res_l1,
        .dual_res = trial_kkt.inf_dual_res,
        .objective = trial_kkt.penalized_obj,
    };

    if (ls.step_cnt == 0 && trial_kkt.prim_res_l1 < current_kkt.prim_res_l1)
        ls.skip_soc = true;

    const scalar_t fullstep_dec = current_kkt.penalized_obj_fullstep_dec;
    ls.alpha_min = current_linesearch_alpha_min(ls);

    // Update best trial
    if (trial_point.prim_res < ls.best_trial.prim_res || trial_point.objective < ls.best_trial.objective) {
        ls.best_trial.prim_res = trial_point.prim_res;
        ls.best_trial.dual_res = trial_point.dual_res;
        ls.best_trial.objective = trial_point.objective;
        ls.best_trial.alpha_primal = settings.ls.alpha_primal;
        ls.best_trial.alpha_dual = settings.ls.alpha_dual;
    }
    if (settings.verbose) {
        fmt::print(
            "[ls] step no. {}, primal res: {:.3e}, search obj: {:.3e}, alpha_primal: {:.3e}, alpha_dual: {:.3e}\n",
            ls.step_cnt, trial_point.prim_res, trial_point.objective, settings.ls.alpha_primal,
            settings.ls.alpha_dual);
        fmt::print("  alpha_min: {:.3e}\n", ls.alpha_min);
        for (size_t i = 0; i < ls.points.size(); ++i) {
            fmt::print("   filter point {}: primal res: {:.3e}, objective: {:.3e}\n",
                       i, ls.points[i].prim_res, ls.points[i].objective);
        }
    }

    const bool accept = ls.try_step(trial_kkt, current_kkt, settings);

    if (accept || ls.stop) {
        ls.recompute_approx = false;
        if (accept && !ls.stop && (!ls.switching_condition || !ls.armijo_cond_met))
            ls.update_filter(current_kkt, settings);
        return accept ? line_search_action::accept : line_search_action::failure;
    }

    ls.recompute_approx = true;
    // try second-order correction (IPOPT §3.4 / A-5.5): only on the first rejection,
    // only when rejected by filter (not Armijo), and κ_soc abort not triggered.
    if (settings.ls.enable_soc && ls.step_cnt == 0 && !ls.skip_soc &&
        ls.soc_iter_cnt < settings.ls.max_soc_iter && !ls.switching_condition) {
    }

    const scalar_t current_primal = current_kkt.prim_res_l1;
    if (settings.ls.alpha_primal <= ls.alpha_min && current_primal > settings.prim_tol) {
        ls.stop = true;
        ls.failure_reason = filter_linesearch_per_iter_data::failure_reason_t::tiny_step;
        ls.recompute_approx = false;
        if (settings.verbose)
            fmt::print("  line search reached min step: alpha_p {:.3e} <= alpha_min {:.3e} with prim_res {:.3e}\n",
                       settings.ls.alpha_primal, ls.alpha_min, current_primal);
        return line_search_action::failure;
    }

    if (settings.ls.max_steps > ls.step_cnt) {
        ls.step_cnt++;
        step_back_alpha(ls);
        if (settings.verbose)
            fmt::print("  backtrack, alpha_p: {:.3e}, alpha_d: {:.3e}\n",
                       settings.ls.alpha_primal, settings.ls.alpha_dual);
        return line_search_action::backtrack;
    }

    // Line search failed — apply fallback strategy
    ls.stop = true;
    ls.failure_reason = filter_linesearch_per_iter_data::failure_reason_t::other;
    if (settings.ls.failure_strategy == linesearch_setting::failure_backup_strategy::min_step) {
        if (settings.verbose) fmt::print("  ls failed, use min primal step...\n");
        ls.enforce_min = true;
        settings.ls.alpha_primal = ls.initial_alpha_primal * std::min(0.01 / ls.initial_alpha_primal, 1.0);
    } else {
        if (settings.verbose) {
            fmt::print("  ls failed, use best trial...\n");
            fmt::print("    best trial primal res: {:.3e}, objective: {:.3e}, alpha_p: {:.3e}, alpha_d: {:.3e}\n",
                       ls.best_trial.prim_res, ls.best_trial.objective,
                       ls.best_trial.alpha_primal, ls.best_trial.alpha_dual);
        }
        settings.ls.alpha_primal = ls.best_trial.alpha_primal;
        if (settings.ls.update_alpha_dual) settings.ls.alpha_dual = ls.best_trial.alpha_dual;
    }
    fmt::println(" line search failed, dec_full_pred = {:.3e}, best trial primal res: {:.3e}, objective: {:.3e}\n",
                 fullstep_dec, ls.best_trial.prim_res, ls.best_trial.objective);
    ls.update_filter(current_kkt, settings);
    return line_search_action::failure;
}

ns_sqp::line_search_action ns_sqp::merit_linesearch(filter_linesearch_data &ls,
                                                     const kkt_info &trial_kkt,
                                                     const kkt_info &current_kkt) {
    const auto merit = [&](scalar_t prim_l1, scalar_t dual_res) -> scalar_t {
        return prim_l1 * prim_l1 + settings.ls.merit_sigma * dual_res * dual_res;
    };

    const scalar_t merit_trial = merit(trial_kkt.prim_res_l1, trial_kkt.avg_dual_res);
    const scalar_t merit_k     = merit(current_kkt.prim_res_l1, current_kkt.avg_dual_res);

    // On the first (full-step) trial, record merit to estimate the directional derivative.
    // dir_deriv ≈ (M(x + 1*d) - M(x)) / 1.0  (finite-difference estimate)
    if (ls.step_cnt == 0) ls.merit_fullstep = merit_trial;

    if (merit_trial < ls.best_merit_trial.merit) {
        ls.best_merit_trial.merit        = merit_trial;
        ls.best_merit_trial.alpha_primal = settings.ls.alpha_primal;
        ls.best_merit_trial.alpha_dual   = settings.ls.alpha_dual;
    }

    if (settings.verbose)
        fmt::print("[merit ls] step {}, merit: {:.3e} (prim: {:.3e}, avg_dual: {:.3e}), alpha_p: {:.3e}, merit_k: {:.3e} (prim: {:.3e}, avg_dual: {:.3e})\n",
                   ls.step_cnt, merit_trial, trial_kkt.prim_res_l1, trial_kkt.avg_dual_res,
                   settings.ls.alpha_primal, merit_k, current_kkt.prim_res_l1, current_kkt.avg_dual_res);

    // Armijo sufficient decrease: M(x + alpha*d) <= M(x) + c * alpha * dir_deriv
    // dir_deriv estimated from the full step (negative when making progress).
    const scalar_t dir_deriv     = ls.merit_fullstep - merit_k;
    const scalar_t armijo_target = merit_k + settings.ls.armijo_dec_frac * settings.ls.alpha_primal * dir_deriv;
    const bool accept            = merit_trial <= armijo_target;

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

    // Line search failed — apply fallback strategy
    ls.stop = true;
    if (settings.ls.failure_strategy == linesearch_setting::failure_backup_strategy::min_step) {
        if (settings.verbose) fmt::print("  merit ls failed, use min primal step...\n");
        ls.enforce_min = true;
        settings.ls.alpha_primal = ls.initial_alpha_primal * std::min(0.01 / ls.initial_alpha_primal, 1.0);
    } else {
        if (settings.verbose)
            fmt::print("  merit ls failed, use best trial (merit: {:.3e}, alpha_p: {:.3e})...\n",
                       ls.best_merit_trial.merit, ls.best_merit_trial.alpha_primal);
        settings.ls.alpha_primal = ls.best_merit_trial.alpha_primal;
        if (settings.ls.update_alpha_dual) settings.ls.alpha_dual = ls.best_merit_trial.alpha_dual;
    }
    fmt::println(" merit line search failed, merit_k: {:.3e}, best merit: {:.3e}\n",
                 merit_k, ls.best_merit_trial.merit);
    return line_search_action::failure;
}

void ns_sqp::second_order_correction() {
}

} // namespace moto
