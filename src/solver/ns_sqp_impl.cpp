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
    // setup solver settings, dont parallelize because settings are shared
    graph_.apply_all_unary_forward([this](solver::data_base *cur) {
        cur->for_each_constr([this, cur](auto &c, auto &d) { dynamic_cast<impl::constr &>(c).setup_solver_setting(d, &settings); });
    });
    // initialize the data
    graph_.apply_all_unary_parallel([this](solver::data_base *cur) {
        cur->update_approximation(true);
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
        settings_t setting_per_thread[n_worker];
        auto finalize_bound_and_set_to_max = [&]() {
            for (size_t i = 0; i < n_worker; ++i) {
                settings.primal.merge_from(setting_per_thread[i].primal);
                settings.dual.merge_from(setting_per_thread[i].dual);
            }
            settings.alpha_primal = settings.primal.alpha_max;
            settings.alpha_dual = settings.dual.alpha_max;
        };
        // timed_block_labeled("all",
        graph_.apply_all_unary_parallel(ns_riccati::ns_factorization);
        graph_.apply_all_binary_forward<true>(ns_riccati::partial_value_derivative);
        graph_.apply_all_binary_backward<true>(ns_riccati::riccati_recursion);
        graph_.apply_all_unary_parallel(ns_riccati::compute_primal_sensitivity);
        graph_.apply_all_binary_forward<false, true>(ns_riccati::fwd_linear_rollout);
        graph_.apply_all_unary_parallel(ns_riccati::finalize_newton_step);
        graph_.apply_all_unary_parallel([&setting_per_thread](size_t tid, auto *d) {
            ineq_soft_solve::calculate_line_search_bounds(d, &setting_per_thread[tid]);
        });
        finalize_bound_and_set_to_max();
        // first round line search
        graph_.apply_all_unary_parallel([this](auto *d) { ns_riccati::line_search_step(d, &settings); });
        // update ipm adaptive mu
        // reset the settings
        if (settings.comp_affine_step()) {
            settings.n_ipm_cstr = 0;
            settings.prev_normalized_comp = 0.;
            settings.after_normalized_comp = 0.;
            for (size_t i = 0; i < n_worker; ++i) {
                settings_t &cfg = setting_per_thread[i];
                settings.prev_normalized_comp += cfg.prev_normalized_comp;
                settings.after_normalized_comp += cfg.after_normalized_comp;
                settings.n_ipm_cstr += cfg.n_ipm_cstr;
            }
            scalar_t eta = settings.after_normalized_comp / (settings.prev_normalized_comp / settings.n_ipm_cstr);
            settings.sig = std::max(0., std::min(1., eta));
            settings.sig = settings.sig * settings.sig * settings.sig; // cubic
            settings.mu = settings.sig * settings.prev_normalized_comp / settings.n_ipm_cstr;
        }

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
