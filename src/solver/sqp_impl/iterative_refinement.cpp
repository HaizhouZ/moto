#include <moto/solver/ns_sqp.hpp>
// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#define SHOW_DETAIL_TIMING

#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/utils/field_conversion.hpp>

namespace moto {
void ns_sqp::iterative_refinement() {
    auto phase_profile = profile_scope(profile_phase::iterative_refinement);
    auto &graph = solver_graph();
    // if (info.inf_prim_step < 1e-1 || info.inf_dual_step < 1e-1) {
    size_t iter_refine_max = settings.rf.max_iters;
    size_t iter_refine = 0;
    detail_timed_block_start("iterative_refinement");
    while (iter_refine < iter_refine_max) {
        {
            auto subphase_profile = profile_scope(profile_phase::iterative_refinement_check_residual);
            detail_timed_block_start("check_residual");
            // finalize the dual step to get the correct dual variables for computing the residual, and compute the residual with the updated dual variables
            graph.for_each_parallel(
                [&](data *d) {
                    riccati_solver_->finalize_dual_newton_step(d);
                    riccati_solver_->compute_kkt_residual(d);
                });
            detail_timed_block_end("check_residual");
        }
        struct MOTO_ALIGN_NO_SHARING inf_res_state_worker {
            scalar_t inf_kkt_stat_err_u = 0.;
            scalar_t inf_kkt_stat_err_y = 0.;
            scalar_t inf_resto_local_stat = 0.;
            scalar_t inf_resto_local_comp = 0.;
        } thread_res[settings.n_worker];
        size_t step = 0;
        graph.apply_forward<true>([&](size_t tid, data *d, data *next) {
            if (d->kkt_stat_err_[__u].size() > 0) {
                thread_res[tid].inf_kkt_stat_err_u = std::max(thread_res[tid].inf_kkt_stat_err_u, d->kkt_stat_err_[__u].cwiseAbs().maxCoeff());
            }
            if (next != nullptr) {
                if (next->kkt_stat_err_[__x].size() > 0) {
                    next->kkt_stat_err_[__x].applyOnTheRight(utils::permutation_from_y_to_x(&d->problem(), &next->problem()));
                    d->kkt_stat_err_[__y] += next->kkt_stat_err_[__x];
                }
            }
            if (d->kkt_stat_err_[__y].size() > 0) {
                thread_res[tid].inf_kkt_stat_err_y = std::max(thread_res[tid].inf_kkt_stat_err_y, d->kkt_stat_err_[__y].cwiseAbs().maxCoeff());
            }
            if (in_restoration_phase()) {
                const auto local_res = solver::restoration::refinement_local_residuals(*d);
                thread_res[tid].inf_resto_local_stat = std::max(thread_res[tid].inf_resto_local_stat, local_res.stationarity);
                thread_res[tid].inf_resto_local_comp = std::max(thread_res[tid].inf_resto_local_comp, local_res.complementarity);
            }
        },
                                   true);
        scalar_t inf_kkt_stat_err_u = 0.;
        scalar_t inf_kkt_stat_err_y = 0.;
        scalar_t inf_resto_local_stat = 0.;
        scalar_t inf_resto_local_comp = 0.;
        for (auto &w : thread_res) {
            inf_kkt_stat_err_u = std::max(inf_kkt_stat_err_u, w.inf_kkt_stat_err_u);
            inf_kkt_stat_err_y = std::max(inf_kkt_stat_err_y, w.inf_kkt_stat_err_y);
            inf_resto_local_stat = std::max(inf_resto_local_stat, w.inf_resto_local_stat);
            inf_resto_local_comp = std::max(inf_resto_local_comp, w.inf_resto_local_comp);
        }
        if (settings.verbose) {
            if (in_restoration_phase()) {
                fmt::print("  iterative refinement {}, kkt_stat_err_u: {:.3e}, kkt_stat_err_y: {:.3e}, resto_local_stat: {:.3e}, resto_local_comp: {:.3e}\n",
                           iter_refine, inf_kkt_stat_err_u, inf_kkt_stat_err_y, inf_resto_local_stat, inf_resto_local_comp);
            } else {
                fmt::print("  iterative refinement {}, kkt_stat_err_u: {:.3e}, kkt_stat_err_y: {:.3e}\n",
                           iter_refine, inf_kkt_stat_err_u, inf_kkt_stat_err_y);
            }
        }
        const scalar_t resto_local_tol = std::max(settings.rf.prim_res_tol, settings.rf.dual_res_tol);
        const bool resto_locals_small =
            !in_restoration_phase() ||
            (inf_resto_local_stat < settings.rf.dual_res_tol && inf_resto_local_comp < resto_local_tol);
        if (inf_kkt_stat_err_u < settings.rf.prim_res_tol &&
            inf_kkt_stat_err_y < settings.rf.dual_res_tol &&
            resto_locals_small) {
            break;
        }
        {
            auto subphase_profile = profile_scope(profile_phase::iterative_refinement_step);
            detail_timed_block_start("iterative_refinement_step");
            run_correction_step(
                [](ns_sqp::data *data) {
                    data->first_order_correction_start([data]() {
                        data->dense().lag_jac_corr_[__u] = data->kkt_stat_err_[__u];
                        data->dense().lag_jac_corr_[__y] = data->kkt_stat_err_[__y];
                    });
                },
                [this](ns_sqp::data *data) {
                    data->first_order_correction_end();
                    finalize_correction(data);
                });
            detail_timed_block_end("iterative_refinement_step");
        }
        iter_refine++;
    }
    detail_timed_block_end("iterative_refinement");
}
} // namespace moto
