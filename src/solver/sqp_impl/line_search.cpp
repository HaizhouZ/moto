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
    if (last_step_was_armijo) {
        return;
    }

    // only add to the filter if the new point is not dominated by the filter
    point current_point = {current_kkt.inf_prim_res, current_kkt.inf_dual_res, current_kkt.objective};
    points.erase(std::remove_if(
                     points.begin(), points.end(),
                     [&](const point &p) {
                         return current_point.dominate(p, settings);
                     }),
                 points.end());
    points.push_back(current_point);
}
bool ns_sqp::filter_linesearch_data::point::dominate(const point &other, const settings_t &settings) const {
    return prim_res <= (1.0 - settings.ls.primal_gamma) * other.prim_res &&
           objective <= other.objective - settings.ls.dual_gamma * other.prim_res;
}
bool ns_sqp::filter_linesearch_data::try_step(const kkt_info &trial_kkt, const kkt_info &current_kkt, settings_t &settings) {

    scalar_t inf_prim_res_trial = trial_kkt.inf_prim_res;
    scalar_t obj_trial = trial_kkt.objective;

    scalar_t inf_prim_res_k = current_kkt.inf_prim_res;
    scalar_t obj_k = current_kkt.objective;

    point trial_point = {inf_prim_res_trial, trial_kkt.inf_dual_res, obj_trial};

    // reject if the trial point is dominated by any point in the filter
    for (const auto &p : points) {
        if (p.dominate(trial_point, settings)) {
            filter_reject_cnt++;
            return false;
        }
    }
    filter_reject_cnt = 0;

    // check switching condition
    bool is_switching = false; // true if in objective decrease mode
    if (current_kkt.obj_ful_step_dec < 0.0 && // note we ignore the other switching condition in IPOPT (too many parameters!)
        inf_prim_res_k <= constr_vio_min) {
        is_switching = true;
    }

    // sufficient objective decrease condition (Armijo) for switching
    if (is_switching) {
        // Armijo condition for the objective
        scalar_t armijo_target = obj_k +
                                 settings.ls.armijo_dec_frac * settings.ls.alpha_primal * current_kkt.obj_ful_step_dec;

        if (obj_trial <= armijo_target) {
            last_step_was_armijo = true;
            return true;
        }
    } else { // filter condition for non-switching
        // Filter condition relative to the CURRENT iterate
        // pass if the trial point is not dominated by the current
        point current_point = {inf_prim_res_k, current_kkt.inf_dual_res, obj_k};
        if (!current_point.dominate(trial_point, settings)) {
            last_step_was_armijo = false;
            return true;
        }
    }

    return false;
}
ns_sqp::line_search_action ns_sqp::filter_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt) {
    if (ls.step_cnt == 0 && trial_kkt.inf_prim_res < current_kkt.inf_prim_res) {
        // skip second-order correction if the first trial already shows improvement in primal residual
        ls.skip_soc = true;
    }
    const auto record_best_trial = [&] {
        bool prim_better = trial_kkt.inf_prim_res < ls.best_trial.prim_res;
        bool obj_better = trial_kkt.objective < ls.best_trial.objective;
        if (prim_better || obj_better) {
            ls.best_trial.prim_res = trial_kkt.inf_prim_res;
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
            fmt::print("    filter point {}: primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}\n", i, ls.points[i].prim_res, ls.points[i].dual_res, ls.points[i].objective);
        }
    };

    record_best_trial();
    if (settings.verbose) {
        fmt::print("  ls step, primal res: {:.3e}, objective: {:.3e}, alpha_primal: {:.3e}\n",
                   trial_kkt.inf_prim_res, trial_kkt.objective, settings.ls.alpha_primal);
    }
    bool accept = ls.try_step(trial_kkt, current_kkt, settings);
    /// if the point is acceptable or we have already tried enough steps, stop line search and accept the point if acceptable
    if (accept || ls.stop) {
        ls.recompute_approx = false;
        if (accept && !ls.stop) {
            ls.update_filter(current_kkt, settings);
        }
        return accept ? line_search_action::accept : line_search_action::stop;
    }

    ls.recompute_approx = true;
    // fmt::print("  previous res, primal res: {:.3e}, dual res: {:.3e}\n", kkt_last.inf_prim_res, kkt_last.inf_dual_res);
    /// try second-order correction before backtracking
    /// skip if line search started already or first trial already shows improvement in primal residual
    if (settings.ls.enable_soc && ls.step_cnt == 0 && !ls.skip_soc &&
        ls.soc_iter_cnt < settings.ls.max_soc_iter) {
        if (settings.verbose)
            fmt::print("  ls retry with second-order correction ({}/{})\n", ls.soc_iter_cnt + 1, settings.ls.max_soc_iter);
        ls.soc_iter_cnt++;
        return line_search_action::retry_second_order_correction;
    }
    print_filter();
    /// backtrack
    if (settings.ls.max_steps > ls.step_cnt) {
        ls.step_cnt++;
        settings.ls.alpha_primal = std::max(
            settings.ls.alpha_primal - ls.initial_alpha_primal / (settings.ls.max_steps + 1e-8),
            scalar_t(0.0));
        if (settings.ls.update_alpha_dual) {
            settings.ls.alpha_dual = std::max(
                settings.ls.alpha_dual - ls.initial_alpha_dual / (settings.ls.max_steps + 1e-8),
                scalar_t(0.0));
        }
        // settings.alpha_primal = (0.7 * settings.alpha_primal - settings.alpha_primal);
        // settings.alpha_dual = -initial_alpha_dual / (max_ls_steps + 1e-8);
        if (settings.verbose)
            fmt::print("  ls backtrack, alpha_p: {:.3e}, alpha_d: {:.3e}\n", settings.ls.alpha_primal, settings.ls.alpha_dual);
        return line_search_action::backtrack;
    }
    // line search failed, fallback to backup
    ls.stop = true; // stop line search, will not try more steps
    if (settings.ls.failure_strategy == linesearch_setting::failure_backup_strategy::min_step) {
        if (settings.verbose) {
            fmt::print("  ls failed, use min primal step and reset ls...\n");
        }
        ls.enforce_min = true;
        auto scale = std::min(0.01 / ls.initial_alpha_primal, 1.0);
        settings.ls.alpha_primal = ls.initial_alpha_primal * scale;
    } else {
        if (settings.verbose) {
            fmt::print("  ls failed, use best trial and reset ls...\n");
            fmt::print("    best trial primal res: {:.3e}, dual res: {:.3e}, objective: {:.3e}, alpha_p: {:.3e}, alpha_d: {:.3e}\n",
                       ls.best_trial.prim_res, ls.best_trial.dual_res, ls.best_trial.objective, ls.best_trial.alpha_primal, ls.best_trial.alpha_dual);
        }
        settings.ls.alpha_primal = ls.best_trial.alpha_primal;
        if (settings.ls.update_alpha_dual) {
            settings.ls.alpha_dual = ls.best_trial.alpha_dual;
        }
    }
    ls.points.clear();
    return line_search_action::stop;
}

void ns_sqp::apply_second_order_correction() {
    graph_.for_each_parallel([](data *d) {
        d->first_order_correction_start([]() {});
    });
    graph_.for_each_parallel(solver_call(&solver_type::ns_factorization_correction));

    correction_step();
    graph_.for_each_parallel([this](data *d) {
        d->first_order_correction_end();
        finalize_correction(d);
    });
}

} // namespace moto