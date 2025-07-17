#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ineq_soft_solve.hpp>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_sqp.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

#define stat_col_width 12

namespace moto {
void ns_sqp::forward() {
    timed_block(graph_.apply_all_unary_parallel(ns_riccati::update_approx));
}
void ns_sqp::update(size_t n_iter) {
    fmt::print("Initialization for SQP...\n");
    graph_.apply_all_unary_parallel([this](solver::data_base *cur) {
        // setup solver settings
        cur->for_each_constr([this](auto &c, auto &d) { c.setup_setting(d, &settings); });
        cur->update_approximation(true);
        // initialize the data
        ineq_soft_solve::initialize(cur);
    });
    std::atomic<double> cost_all{0.};
    graph_.apply_all_unary_parallel([&cost_all](auto *n) {
        cost_all += n->dense_->cost_;
    });
    fmt::print("initial cost_total: {}\n", cost_all.load());
    graph_.apply_all_unary_parallel(ns_riccati::update_approx);

    // print statistics header
    constexpr std::string_view terms[] = {"objective", "inf_prim_res", "inf_dual_res", "alpha_primal", "alpha_dual"};
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
        size_t n_worker = get_num_threads();
        settings_t::worker setting_per_thread[n_worker];
        auto finalize_bound_and_set_to_max = [&]() {
            for (size_t i = 0; i < n_worker; ++i) {
                settings.primal.merge_from(setting_per_thread[i].primal);
                settings.dual.merge_from(setting_per_thread[i].dual);
            }
            settings.alpha_primal = settings.primal.alpha_max;
            settings.alpha_dual = settings.dual.alpha_max;
            // // copy the settings to each worker
            // for (size_t i = 0; i < n_worker; ++i) {
            //     setting_per_thread[i].primal = settings.primal;
            //     setting_per_thread[i].dual = settings.dual;
            //     setting_per_thread[i].alpha_primal = settings.alpha_primal;
            //     setting_per_thread[i].alpha_dual = settings.alpha_dual;
            // }
        };
        // timed_block_labeled("all",
        graph_.apply_all_unary_parallel(ns_riccati::ns_factorization);
        graph_.apply_all_binary_backward<true>(ns_riccati::riccati_recursion);
        graph_.apply_all_unary_parallel(ns_riccati::compute_primal_sensitivity);
        graph_.apply_all_binary_forward<false, true>(ns_riccati::fwd_linear_rollout);

        if (settings.ipm_compute_affine_step()) // compute the affine step, no need to finalize dual step
            graph_.apply_all_unary_parallel([](auto *d) { ns_riccati::finalize_newton_step(d, false); });
        else // directly finalize the dual step
            graph_.apply_all_unary_parallel([](auto *d) { ns_riccati::finalize_newton_step(d, true); });
        // decide line search bounds (e.g., fraction-to-bounds)
        graph_.apply_all_unary_parallel([&setting_per_thread](size_t tid, auto *d) {
            ineq_soft_solve::calculate_line_search_bounds(d, &setting_per_thread[tid]);
        });
        finalize_bound_and_set_to_max();
        if (settings.ipm_compute_affine_step()) {
            // line search with max bounds
            graph_.apply_all_unary_parallel([&setting_per_thread](size_t tid, auto *d) {
                ineq_soft_solve::line_search_step(d, &setting_per_thread[tid]);
            });
            // collect worker ipm data
            auto &main_worker = setting_per_thread[0];
            for (size_t i = 1; i < n_worker; ++i) {
                auto &cfg = setting_per_thread[i];
                main_worker.prev_normalized_comp += cfg.prev_normalized_comp;
                main_worker.after_normalized_comp += cfg.after_normalized_comp;
                main_worker.n_ipm_cstr += cfg.n_ipm_cstr;
            }
            // adaptive mu update
            // eta = after / before
            scalar_t eta = main_worker.after_normalized_comp / main_worker.prev_normalized_comp;
            settings.sig = std::max(0., std::min(1., eta));            // clip
            settings.sig = settings.sig * settings.sig * settings.sig; // cubic
            settings.mu = settings.sig * main_worker.prev_normalized_comp / main_worker.n_ipm_cstr;
            // use the new mu to update the rhs jacobian
            graph_.apply_all_unary_parallel(solver::prepare_correction);
            graph_.apply_all_unary_parallel(ineq_soft_solve::for_each([](impl::soft_constr &c, auto &d) {
                c.correct_jacobian(d);
            }));
            // solve the problem again with updated mu
            graph_.apply_all_binary_backward<true>(ns_riccati::riccati_recursion_correction);
            graph_.apply_all_unary_parallel(ns_riccati::compute_primal_sensitivity_correction);
            graph_.apply_all_binary_forward<false, true>(ns_riccati::fwd_linear_rollout_correction);
            graph_.apply_all_unary_parallel(ns_riccati::finalize_newton_step_correction);
        }
        /// @todo: update the line search stepsize?
        // real line search step
        graph_.apply_all_unary_parallel([this](auto *d) { ns_riccati::line_search_step(d, &settings); });

        graph_.apply_all_unary_parallel(ns_riccati::update_approx);
        // );
        kkt_info info;
        for (auto &n : graph_.get_unordered_flattened_nodes()) {
            info.objective += n->cost();
            info.inf_prim_res = std::max(n->inf_prim_res(), info.inf_prim_res);
            info.inf_dual_res = std::max(info.inf_dual_res, n->dense_->jac_[__u].cwiseAbs().maxCoeff());
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
        scalar_t stats[] = {info.objective, info.inf_prim_res, info.inf_dual_res,
                            settings.alpha_primal, settings.alpha_dual};
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
