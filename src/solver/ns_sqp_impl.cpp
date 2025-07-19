#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_sqp.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

#define stat_col_width 15

namespace moto {
void ns_sqp::forward() {
    timed_block(graph_.apply_all_unary_parallel(solver::ns_riccati::update_approx));
}
void ns_sqp::update(size_t n_iter) {
    fmt::print("Initialization for SQP...\n");
    graph_.apply_all_unary_parallel([this](solver::data_base *cur) {
        // setup solver settings
        cur->for_each_constr([this](auto &c, auto &d) { c.setup_workspace_data(d, &settings); });
        cur->update_approximation(true);
        // initialize the data
        solver::ineq_soft::initialize(cur);
    });
    std::atomic<double> cost_all{0.};
    graph_.apply_all_unary_parallel([&cost_all](auto *n) {
        cost_all += n->dense_->cost_;
    });
    fmt::print("initial cost_total: {}\n", cost_all.load());
    graph_.apply_all_unary_parallel(solver::ns_riccati::update_approx);

    // print statistics header
    constexpr std::string_view terms[] = {"objective", "inf_prim_res", "inf_dual_res", "inf_comp_res", "alpha_primal", "alpha_dual", "ipm_mu"};
    size_t total_length = 4 + std::size(terms) * (stat_col_width + 4) + 1;
    fmt::print("{:-<{}}\n", "", total_length);
    fmt::print("no. |");
    for (const auto &term : terms) {
        fmt::print("| {:<{}} |", term, stat_col_width);
    }
    fmt::print("\n");
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// main loop
    for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
        settings.ls_config_reset();
        size_t n_worker = get_num_threads();
        settings_t::worker setting_per_thread[n_worker];
        auto finalize_bound_and_set_to_max = [&]() {
            for (size_t i : range(n_worker)) {
                settings.primal.merge_from(setting_per_thread[i].primal);
                settings.dual.merge_from(setting_per_thread[i].dual);
            }
            settings.alpha_primal = settings.primal.alpha_max;
            settings.alpha_dual = settings.dual.alpha_max;
            // copy the settings to each worker
            for (size_t i : range(n_worker)) {
                setting_per_thread[i].copy_from(settings);
            }
        };
        // timed_block_labeled("all",
        graph_.apply_all_unary_parallel(solver::ns_riccati::ns_factorization);
        graph_.apply_all_binary_backward<true>(solver::ns_riccati::riccati_recursion);
        graph_.apply_all_unary_parallel(solver::ns_riccati::compute_primal_sensitivity);
        graph_.apply_all_binary_forward<false, true>(solver::ns_riccati::fwd_linear_rollout);

        bool finalize_dual = true;
        if (settings.ipm_enable_affine_step()) { // compute the affine step, no need to finalize dual step
            settings.ipm_start_predictor_computation();
            finalize_dual = false; // do not finalize dual step
        }
        graph_.apply_all_unary_parallel([finalize_dual, &setting_per_thread](size_t tid, auto *d) {
            solver::ns_riccati::finalize_newton_step(d, finalize_dual);
            solver::ineq_soft::finalize_newton_step(d);
            // decide line search bounds (e.g., fraction-to-bounds)
            solver::ineq_soft::calculate_line_search_bounds(d, &setting_per_thread[tid]);
        });
        finalize_bound_and_set_to_max();
        if (settings.ipm_enable_affine_step()) {
            // line search with max bounds
            graph_.apply_all_unary_parallel([&setting_per_thread](size_t tid, auto *d) {
                solver::ineq_soft::finalize_predictor_step(d, &setting_per_thread[tid]);
            });
            settings.ipm_end_predictor_computation(); // ipm affine step computation is done
            // collect worker ipm data
            solver::ipm_config::worker &main_worker = setting_per_thread[0];
            for (size_t i : range(n_worker)) {
                main_worker += setting_per_thread[i];
            }
            // adaptive mu update
            settings.adaptive_mu_update(main_worker);
            // use the new mu to update the rhs jacobian
            graph_.apply_all_unary_parallel(solver::ineq_soft::first_order_correction_start);
            // solve the problem again with updated mu
            graph_.apply_all_binary_backward<true>(solver::ns_riccati::riccati_recursion_correction);
            graph_.apply_all_unary_parallel(solver::ns_riccati::compute_primal_sensitivity_correction);
            graph_.apply_all_binary_forward<false, true>(solver::ns_riccati::fwd_linear_rollout_correction);
            graph_.apply_all_unary_parallel([](auto *d) {
                solver::ineq_soft::first_order_correction_end(d);
                solver::ns_riccati::finalize_newton_step_correction(d);
                solver::ineq_soft::finalize_newton_step(d);
            });
            // recompute line search bounds with the corrected newton step
            settings.ls_config_reset();
            for (size_t i : range(n_worker)) {
                setting_per_thread[i].ls_config_reset();
            }
            graph_.apply_all_unary_parallel([&setting_per_thread](size_t tid, auto *d) {
                solver::ineq_soft::calculate_line_search_bounds(d, &setting_per_thread[tid]);
            });
            finalize_bound_and_set_to_max();
        }
        /// @todo: update the line search stepsize?
        // real line search step
        graph_.apply_all_unary_parallel([this](auto *d) {
            solver::ns_riccati::line_search_step(d, &settings);
            solver::ineq_soft::line_search_step(d, &settings);
        });

        graph_.apply_all_unary_parallel(solver::ns_riccati::update_approx);
        // );
        kkt_info info;
        for (auto &n : graph_.get_unordered_flattened_nodes()) {
            info.objective += n->cost();
            info.inf_prim_res = std::max(info.inf_prim_res, n->inf_prim_res_);
            info.inf_dual_res = std::max(info.inf_dual_res, n->dense_->jac_[__u].cwiseAbs().maxCoeff());
            info.inf_comp_res = std::max(info.inf_comp_res, n->inf_comp_res_);
        }
        graph_.apply_all_binary_forward<false, true>([&info](node_data *cur, node_data *next) {
            if (next != nullptr) [[likely]] {
                // cancellation of jacobian from y to x
                static row_vector tmp;
                tmp.conservativeResize(next->dense_->jac_[__x].cols());
                tmp.noalias() = next->dense_->jac_[__x] * permutation_from_y_to_x(cur->prob_, next->prob_) + cur->dense_->jac_[__y];
                info.inf_dual_res = std::max(info.inf_dual_res, tmp.cwiseAbs().maxCoeff());
            } else /// @todo: include initial jac[__x] inf norm if init is optimized
                info.inf_dual_res = std::max(info.inf_dual_res, cur->dense_->jac_[__y].cwiseAbs().maxCoeff());
        });
        // print statistics
        scalar_t stats[] = {info.objective, info.inf_prim_res, info.inf_dual_res, info.inf_comp_res,
                            settings.alpha_primal, settings.alpha_dual, settings.mu};
        fmt::print("{:<3} |", i_iter);
        for (const auto &stat : stats) {
            fmt::print("| {:<{}.6e} |", stat, stat_col_width);
        }
        fmt::print("\n");

        // });
        moto::utils::timing_storage<"all">::get().count = n_iter;
    }
}
} // namespace moto
