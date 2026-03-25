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

void ns_sqp::filter_linesearch_data::update_filter(scalar_t prim_new, scalar_t dual_new, settings_t &settings) {
    // only add to the filter if the new point is not dominated by the filter
    points.erase(std::remove_if(points.begin(), points.end(),
                                [&](const point &p) {
                                    return prim_new <= (1 - settings.ls.primal_gamma) * p.prim_res &&
                                           dual_new <= p.dual_res - settings.ls.dual_gamma * p.prim_res;
                                }),
                 points.end());
    points.push_back({prim_new, dual_new});
}
bool ns_sqp::filter_linesearch_data::try_step(scalar_t prim_res, scalar_t dual_res, settings_t &settings, scalar_t alpha) {

    // A point is acceptable only if it is NOT dominated by ANY point in the filter
    for (const auto &p : points) {
        // Check if the trial point falls into the prohibited region of point 'p'
        // Using Equation (22): Prohibited if (Worse Prim AND Worse Dual)
        bool worse_prim = prim_res >= (1 - settings.ls.primal_gamma) * p.prim_res;
        bool worse_dual = dual_res >= p.dual_res - settings.ls.dual_gamma * p.prim_res; // γ_φ * θ(x_k)
        // bool worse_dual = dual_res >= (1 - settings.ls.dual_gamma) * p.dual_res; // γ_φ * θ(x_k)

        if (prim_res <= settings.prim_tol && settings.ls.enable_dual_cut) { // if primal residual is small, be more lenient on dual residual
            // worse_dual |= dual_res >= settings.ls.dual_cut_coeff * p.dual_res;
            scalar_t elastic_coeff = 1.0 - settings.ls.eta * alpha;
            bool failed_dual_armijo = dual_res >= elastic_coeff * p.dual_res;
            if (failed_dual_armijo) {
                if (settings.verbose) {
                    fmt::print("small prim res, dual_res: {:.3e}, p.dual_res: {:.3e}\n", dual_res, p.dual_res);
                    fmt::print("    worse dual because of small prim res\n");
                }
                return false;
            }
        }

        if (worse_prim && worse_dual) {
            return false; // Point is dominated/prohibited by 'p'!
        }
    }
    // update_filter(prim_res, dual_res, settings);
    return true;
}
void ns_sqp::filter_linesearch(filter_linesearch_data &ls, const kkt_info &kkt) {
    // if (!ls.stop && kkt.inf_dual_res > 0.9999 * kkt_last.inf_dual_res &&
    // kkt.inf_prim_res > 0.9999 * kkt_last.inf_prim_res) {

    const auto record_best_trial = [&] {
        if (kkt.inf_prim_res < ls.best_trial.prim_res || kkt.inf_dual_res < ls.best_trial.dual_res) {
            ls.best_trial.prim_res = kkt.inf_prim_res;
            ls.best_trial.dual_res = kkt.inf_dual_res;
            ls.best_trial.alpha_primal = settings.ls.alpha_primal;
            ls.best_trial.alpha_dual = settings.ls.alpha_dual;
        }
    };
    const auto print_filter = [&] {
        if (!settings.verbose)
            return;
        for (size_t i = 0; i < ls.points.size(); ++i) {
            fmt::print("    filter point {}: primal res: {:.3e}, dual res: {:.3e}\n", i, ls.points[i].prim_res, ls.points[i].dual_res);
        }
    };

    record_best_trial();
    if (settings.verbose) {
        fmt::print("  ls step, primal res: {:.3e}, dual res: {:.3e}, alpha_primal: {:.3e}\n", kkt.inf_prim_res, kkt.inf_dual_res, settings.ls.alpha_primal);
    }
    const bool accept = ls.try_step(kkt.inf_prim_res, kkt.inf_dual_res, settings, settings.ls.alpha_primal);

    if (accept || ls.stop) {
        ls.recompute_approx = false;
        if (accept && !ls.stop) {
            ls.points.clear(); // clear the filter if the switching condition is met, since we are essentially starting a new line search with a different approximation
            ls.update_filter(kkt.inf_prim_res, kkt.inf_dual_res, settings);
        }
        return;
    }

    ls.recompute_approx = true;
    print_filter();
    // fmt::print("  previous res, primal res: {:.3e}, dual res: {:.3e}\n", kkt_last.inf_prim_res, kkt_last.inf_dual_res);
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
        return;
    }

    ls.stop = true;
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
            fmt::print("    best trial primal res: {:.3e}, dual res: {:.3e}, alpha_p: {:.3e}, alpha_d: {:.3e}\n",
                       ls.best_trial.prim_res, ls.best_trial.dual_res, ls.best_trial.alpha_primal, ls.best_trial.alpha_dual);
        }
        settings.ls.alpha_primal = ls.best_trial.alpha_primal;
        if (settings.ls.update_alpha_dual) {
            settings.ls.alpha_dual = ls.best_trial.alpha_dual;
        }
    }
    // auto scale = std::min(0.01 / std::max(initial_alpha_primal, initial_alpha_dual), 1.0);
    // auto scale2 = std::min(0.01 / ls.initial_alpha_dual, 1.0);
    /// @todo: hard constraints use alpha_primal, ipm uses alpha_dual with a backup of multipliers
    // settings.mu = settings.ipm.mu0; // reset mu to initial value when line search fails
    // settings.mu = settings.worker_data.post_aff_comp / settings.worker_data.n_ipm_cstr; // reset mu to the complementarity after affine step when line search fails, this is more aggressive than resetting to mu0 and can help reduce the number of iterations
    // graph_.for_each_parallel(solver::ineq_soft::initialize);
    ls.points.clear();
    ls.recompute_approx = true;
}

} // namespace moto