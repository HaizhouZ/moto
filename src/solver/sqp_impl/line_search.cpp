#include <moto/solver/ns_sqp.hpp>
// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#define SHOW_DETAIL_TIMING
#include <moto/solver/ineq_soft.hpp>

namespace moto {

void ns_sqp::finalize_ls_bound_and_set_to_max() {
    // merge line search bounds from each thread
    for (solver::linesearch_config &s : setting_per_thread) {
        settings.primal.merge_from(s.primal);
        settings.dual.merge_from(s.dual);
    }
    settings.alpha_primal = settings.primal.alpha_max;
    settings.alpha_dual = settings.dual.alpha_max;
    // copy the settings to each worker
    for (solver::linesearch_config &s : setting_per_thread) {
        s.copy_from(settings);
    }
}

void ns_sqp::backtrack_linesearch(ls_info &ls, const kkt_info &kkt) {
    if (!ls.stop && kkt.inf_dual_res > 0.9999 * kkt_last.inf_dual_res &&
        kkt.inf_prim_res > 0.9999 * kkt_last.inf_prim_res) {
        ls.recompute_approx = true;
        // if (settings.verbose) {
        //     fmt::print("  ls step, primal res: {:.3e}, dual res: {:.3e}\n", kkt.inf_prim_res, kkt.inf_dual_res);
        //     fmt::print("  previous res, primal res: {:.3e}, dual res: {:.3e}\n", kkt_last.inf_prim_res, kkt_last.inf_dual_res);
        // }
        if (settings.max_ls_steps > ls.step_cnt) {
            ls.step_cnt++;
            settings.alpha_primal = -ls.initial_alpha_primal / (settings.max_ls_steps + 1e-8);
            // settings.alpha_dual = -initial_alpha_dual / (max_ls_steps + 1e-8);
            if (settings.verbose)
                fmt::print("  ls backtrack, alpha_p: {:.3e}, alpha_d: {:.3e}\n", settings.alpha_primal, settings.alpha_dual);
        } else {
            ls.stop = true;
            ls.enforce_min = true;
            // auto scale = std::min(0.01 / std::max(initial_alpha_primal, initial_alpha_dual), 1.0);
            auto scale = std::min(0.01 / ls.initial_alpha_primal, 1.0);
            // auto scale2 = std::min(0.01 / ls.initial_alpha_dual, 1.0);
            if (settings.verbose) {
                fmt::print("  ls failed, use min primal step and reset ipm...\n");
                // fmt::print("  scale: {:.3e}\n", scale);
                // fmt::print("  initial_alpha_primal: {:.3e}, initial_alpha_dual: {:.3e}\n", ls.initial_alpha_primal, ls.initial_alpha_dual);
            }
            /// @warning this works only because effective alpha_primal is already 0 (max steps taken)
            settings.alpha_primal = ls.initial_alpha_primal * scale;
            // settings.alpha_dual = initial_alpha_dual * scale2;
            settings.alpha_dual = 0.0;
            settings.mu = settings.mu0;
            graph_.for_each_parallel(solver::ineq_soft::initialize);
        }
    } else {
        ls.recompute_approx = false;
        if (!ls.enforce_min) { /// get the real effective step size
            settings.alpha_primal = ls.initial_alpha_primal - ls.initial_alpha_primal / (settings.max_ls_steps + 1e-8) * ls.step_cnt;
        }
        // settings.alpha_dual = initial_alpha_dual - initial_alpha_dual / (max_ls_steps + 1e-8) * step_cnt;
    }
}

} // namespace moto