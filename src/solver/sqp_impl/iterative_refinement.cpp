#include <moto/solver/ns_sqp.hpp>
// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#define SHOW_DETAIL_TIMING

#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/utils/field_conversion.hpp>

namespace moto {
void ns_sqp::iterative_refinement() {
    // if (info.inf_prim_step < 1e-1 || info.inf_dual_step < 1e-1) {
    size_t iter_refine_max = settings.rf.max_iters;
    size_t iter_refine = 0;
    detail_timed_block_start("iterative_refinement");
    while (iter_refine < iter_refine_max) {
        detail_timed_block_start("check_residual");
        graph_.for_each_parallel(bind(&solver_type::compute_kkt_residual));
        detail_timed_block_end("check_residual");
        struct MOTO_ALIGN_NO_SHARING inf_res_state_worker {
            scalar_t inf_res_stat_u = 0.;
            scalar_t inf_res_stat_y = 0.;
        } thread_res[settings.n_worker];
        size_t step = 0;
        graph_.apply_forward<true>([&](size_t tid, data *d, data *next) {
            thread_res[tid].inf_res_stat_u = std::max(thread_res[tid].inf_res_stat_u, d->dense().res_stat_[__u].cwiseAbs().maxCoeff());
            if (next != nullptr) {
                next->dense().res_stat_[__x].applyOnTheRight(utils::permutation_from_y_to_x(&d->problem(), &next->problem()));
                d->dense().res_stat_[__y] += next->dense().res_stat_[__x];
            }
            thread_res[tid].inf_res_stat_y = std::max(thread_res[tid].inf_res_stat_y, d->dense().res_stat_[__y].cwiseAbs().maxCoeff());
        },
                                   true);
        scalar_t inf_res_stat_u = 0.;
        scalar_t inf_res_stat_y = 0.;
        for (auto &w : thread_res) {
            inf_res_stat_u = std::max(inf_res_stat_u, w.inf_res_stat_u);
            inf_res_stat_y = std::max(inf_res_stat_y, w.inf_res_stat_y);
        }
        if (settings.verbose)
            fmt::print("  iterative refinement {}, res_stat_u: {:.3e}, res_stat_y: {:.3e}\n",
                       iter_refine, inf_res_stat_u, inf_res_stat_y);
        if (inf_res_stat_u < settings.rf.prim_res_tol && inf_res_stat_y < settings.rf.dual_res_tol) {
            break;
        }
        detail_timed_block_start("iterative_refinement_step");
        detail_timed_block_start("iterative_refinement_presolve");
        // prepare for iterative refinement
        graph_.for_each_parallel([this](ns_sqp::data *data) {
            data->first_order_correction_start([data]() {
                data->dense().jac_modification_[__u] = data->dense().res_stat_[__u];
                data->dense().jac_modification_[__y] = data->dense().res_stat_[__y];
            });
        });
        detail_timed_block_end("iterative_refinement_presolve");
        correction_step();
        detail_timed_block_start("iterative_refinement_step_finalize");
        // end iterative refinement
        graph_.for_each_parallel([this](ns_sqp::data *data) {
            data->first_order_correction_end();
            finalize_correction(data);
        });
        detail_timed_block_end("iterative_refinement_step_finalize");
        detail_timed_block_end("iterative_refinement_step");
        // recompute line search bounds with the corrected newton step
        settings.as<solver::linesearch_config>().reset();
        for (solver::linesearch_config &s : setting_per_thread) {
            s.reset();
        }
        graph_.for_each_parallel([this](size_t tid, data *d) {
            solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
        });
        finalize_ls_bound_and_set_to_max();
        iter_refine++;
    }
    detail_timed_block_end("iterative_refinement");
}
} // namespace moto