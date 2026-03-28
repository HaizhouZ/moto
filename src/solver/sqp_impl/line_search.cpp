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
    point current_point = {current_kkt.prim_res_l1, current_kkt.inf_dual_res,
                           current_kkt.cost - settings.ipm.mu * current_kkt.log_slack_sum};
    points.erase(std::remove_if(
                     points.begin(), points.end(),
                     [&](const point &p) {
                         return p.in_filter(current_point, settings);
                     }),
                 points.end());
    fmt::print("  added point to filter: primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n",
               current_point.prim_res, current_point.dual_res, current_point.objective);
    points.push_back(current_point);
}
bool ns_sqp::filter_linesearch_data::point::in_filter(const point &filter_entry, const settings_t &settings) const {
    return prim_res >= (1.0 - settings.ls.primal_gamma) * filter_entry.prim_res and
           objective >= filter_entry.objective - settings.ls.dual_gamma * filter_entry.prim_res;
}
bool ns_sqp::filter_linesearch_data::try_step(const kkt_info &trial_kkt, const kkt_info &current_kkt, settings_t &settings) {

    scalar_t prim_res_trial = trial_kkt.prim_res_l1;
    scalar_t obj_trial = trial_kkt.objective; // always fresh: cost - mu * log_slack_sum with current mu

    scalar_t prim_res_k = current_kkt.prim_res_l1;
    // recompute with current mu in case mu changed after current_kkt was computed
    scalar_t mu = settings.ipm.mu;
    scalar_t obj_k = current_kkt.cost - mu * current_kkt.log_slack_sum;
    scalar_t fullstep_dec_k = current_kkt.obj_fullstep_dec - mu * current_kkt.barrier_dir_deriv;

    point trial_point = {prim_res_trial, trial_kkt.inf_dual_res, obj_trial};

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
        if (!switching_condition) {
            fmt::print("  cost step dec: {:.3e}, full step decrease: {:.3e}, switching lhs: {:.3e}, switching rhs: {:.3e}\n",
                       current_kkt.obj_fullstep_dec, fullstep_dec_k, settings.ls.alpha_primal * std::pow(-fullstep_dec_k, settings.ls.s_phi), std::pow(current_kkt.prim_res_l1, settings.ls.s_theta));
        }
    }
    // reject if the trial point is dominated by any point in the filter
    for (const auto &p : points) {
        if (trial_point.in_filter(p, settings)) {
            filter_reject_cnt++;
            if (settings.verbose) {
                fmt::print("  trial point rejected by filter (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}), dominated by filter point (primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e})\n",
                           trial_point.prim_res, trial_point.dual_res, trial_point.objective,
                           p.prim_res, p.dual_res, p.objective);
            }
            return false;
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
            return false;
        }
    } else { // filter condition for non-switching
        // Filter condition relative to the CURRENT iterate
        // pass if the trial point makes sufficient progress in either primal residual or objective compared to the current point (not the filter points)
        point current_point = {prim_res_k, current_kkt.inf_dual_res, obj_k};
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
            return false;
        }
    }
}
ns_sqp::line_search_action ns_sqp::filter_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt) {
    if (ls.step_cnt == 0 && trial_kkt.prim_res_l1 < current_kkt.prim_res_l1) {
        // skip second-order correction if the first trial already shows improvement in primal residual
        ls.skip_soc = true;
    }
    const auto record_best_trial = [&] {
        bool prim_better = trial_kkt.prim_res_l1 < ls.best_trial.prim_res;
        bool obj_better = trial_kkt.objective < ls.best_trial.objective;
        if (prim_better || obj_better) {
            ls.best_trial.prim_res = trial_kkt.prim_res_l1;
            ls.best_trial.dual_res = trial_kkt.inf_dual_res;
            ls.best_trial.objective = trial_kkt.objective;
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
    }
    print_filter();
    bool accept = ls.try_step(trial_kkt, current_kkt, settings);
    /// if the point is acceptable or we have already tried enough steps, stop line search and accept the point if acceptable
    ls.points.clear();
    accept = true;
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
        return accept ? line_search_action::accept : line_search_action::stop;
    }

    ls.recompute_approx = true;
    // try second-order correction (IPOPT §3.4 / A-5.5): only on the first rejection,
    // only when rejected by filter (not Armijo), and κ_soc abort not triggered.
    if (settings.ls.enable_soc && ls.step_cnt == 0 && !ls.skip_soc &&
        ls.soc_iter_cnt < settings.ls.max_soc_iter && !ls.switching_condition) {
        
    }
    /// backtrack
    if (settings.ls.max_steps > ls.step_cnt) {
        ls.step_cnt++;
        settings.ls.alpha_primal = std::max(
            settings.ls.alpha_primal - ls.initial_alpha_primal / (settings.ls.max_steps + 1e-8),
            scalar_t(0.0));
        // if (settings.ls.update_alpha_dual) {
        //     settings.ls.alpha_dual = std::max(
        //         settings.ls.alpha_dual - ls.initial_alpha_dual / (settings.ls.max_steps + 1e-8),
        //         scalar_t(0.0));
        // }
        if (settings.verbose)
            fmt::print("  backtrack, alpha_p: {:.3e}, alpha_d: {:.3e}\n", settings.ls.alpha_primal, settings.ls.alpha_dual);
        return line_search_action::backtrack;
    }
    // line search failed, fallback to backup
    ls.stop = true; // stop line search, will not try more steps
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

    // fails, will go to restoration
    fmt::println(" line search failed, dec_full_pred = {:.3e}, best trial primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n",
                 current_kkt.obj_fullstep_dec, ls.best_trial.prim_res, ls.best_trial.dual_res, ls.best_trial.objective);
    ls.update_filter(current_kkt, settings);
    return line_search_action::stop;
}

void ns_sqp::second_order_correction() {
}

} // namespace moto