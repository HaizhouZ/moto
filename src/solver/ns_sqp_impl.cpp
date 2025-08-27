#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/field_conversion.hpp>
#include <numeric>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

// #define SHOW_DETAIL_TIMING

#ifdef SHOW_DETAIL_TIMING
#define detail_timed_block_start(label) timed_block_start(label)
#define detail_timed_block_end(label) timed_block_end(label)
#else
#define detail_timed_block_start(label)
#define detail_timed_block_end(label)
#endif

#define stat_width 15

namespace moto {
void ns_sqp::forward() {
    graph_.for_each_parallel(data::update_approx);
}

struct stat_item {
    std::string_view name;
    size_t width; // default width for each stat item
    stat_item(std::string_view n, size_t w = stat_width) : name(n), width(w) {}
    void print_header() const {
        fmt::print("| {:<{}} |", name, width);
    }
};

stat_item stats[] = {{"no.", 3},
                     {"objective"},
                     {"prim_res"},
                     {"dual_res"},
                     {"comp_res"},
                     {"||p||"},
                     {"||d||"},
                     {"alpha_p"},
                     {"alpha_d"},
                     {"ipm_mu", stat_width + 2}};

void ns_sqp::print_stats(int i_iter, const kkt_info &info, bool has_ineq) {
    scalar_t stats_value[] = {i_iter, info.objective, info.inf_prim_res, info.inf_dual_res, info.inf_comp_res, info.inf_prim_step, info.inf_dual_step,
                              settings.alpha_primal, settings.alpha_dual, settings.mu};
    std::string_view ipm_flags;
    if (has_ineq && settings.ipm_enable_corrector()) {
        if (settings.ipm_accept_corrector()) {
            ipm_flags = "[c:a]";
        } else {
            ipm_flags = "[c:r]";
        }
    }
    size_t idx_stat = 0;
    for (auto &item : stats) {
        if (item.name == "no.") {
            fmt::print("| {:<{}} |", i_iter < 0 ? "--" : std::to_string(i_iter), item.width);
        } else if (item.name == "ipm_mu") {
            fmt::print("| {:<{}} |", has_ineq ? fmt::format("{:.6e}{}", stats_value[idx_stat], ipm_flags) : "---------", item.width);
        } else {
            fmt::print("| {:<{}.6e} |", stats_value[idx_stat], item.width);
        }
        idx_stat++;
    }

    fmt::print("\n");
};
void iterative_refinement_start(ns_sqp::data *data) {
    data->Q_y_corr = nullptr;
    data->prim_corr[__x].setZero();
    // data->clear_merit_jac();
    // clear modification
    for (auto field : primal_fields) {
        data->dense().jac_modification_[field].setZero();
    }
    /// @todo fill the residual here
    data->dense().jac_modification_[__u] = data->dense().res_stat_[__u];
    data->dense().jac_modification_[__y] = data->dense().res_stat_[__y];
    data->swap_jacobian_modification(); // move modification to the jacobian for later solving
}
void iterative_refinement_end(ns_sqp::data *data) {
    data->swap_jacobian_modification();                     // move
    data->Q_y_corr = &data->dense().jac_modification_[__y]; // cache the Q_y after correction
    solver::ns_riccati::finalize_newton_step_correction(data);
    solver::ineq_soft::finalize_newton_step(data, false);
    solver::ns_riccati::finalize_dual_newton_step(data);
}

void ns_sqp::update(size_t n_iter) {
    fmt::print("Initialization for SQP...\n");
    graph_.for_each_parallel([this](data *cur) {
        // setup solver settings
        cur->for_each_constr([this](const generic_func &c, func_approx_data &d) { c.setup_workspace_data(d, &settings); });
        cur->update_approximation(node_data::update_mode::eval_val);
        // initialize the data
        solver::ineq_soft::initialize(cur);
    });
    graph_.for_each_parallel([](data *cur) {
        cur->update_approximation(node_data::update_mode::eval_derivatives);
    });

    // print statistics header
    size_t widths[] = {3, stat_width, stat_width, stat_width, stat_width, stat_width, stat_width, stat_width};
    for (const auto &term : stats) {
        term.print_header();
    }
    fmt::print("\n");
    print_stats(-1, compute_kkt_info(), false); // print initial stats
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// main loop
    for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
        bool has_ineq = false;
        for (data &n : graph_.nodes()) {
            if (n.problem().dim(__ineq_x) > 0 || n.problem().dim(__ineq_xu) > 0) {
                has_ineq = true;
                break;
            }
        }
        settings.ls_config_reset();
        size_t n_worker = graph_.n_jobs();
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

        timed_block_start("sqp_single_iter");
        detail_timed_block_start("ns factorization");
        graph_.for_each_parallel(solver::ns_riccati::ns_factorization);
        detail_timed_block_end("ns factorization");

        detail_timed_block_start("riccati_recursion");
        graph_.apply_backward(solver::ns_riccati::riccati_recursion, true);
        detail_timed_block_end("riccati_recursion");
        detail_timed_block_start("post solve");
        graph_.for_each_parallel(solver::ns_riccati::compute_primal_sensitivity);
        detail_timed_block_end("post solve");
        detail_timed_block_start("fwd_linear_rollout");
        graph_.apply_forward(solver::ns_riccati::fwd_linear_rollout, true);
        detail_timed_block_end("fwd_linear_rollout");

        bool finalize_dual = true;
        bool update_res_stat = true;

        if (has_ineq && settings.ipm_enable_affine_step()) { // compute the affine step, no need to finalize dual step
            settings.ipm_start_predictor_computation();
            finalize_dual = false;   // do not finalize dual step
            update_res_stat = false; // do not update stationary residual
        }
        detail_timed_block_start("finalize_newton_step");
        graph_.for_each_parallel([finalize_dual, &setting_per_thread](size_t tid, data *d) {
            solver::ns_riccati::finalize_newton_step(d, finalize_dual);
            solver::ineq_soft::finalize_newton_step(d, false);
            // decide line search bounds (e.g., fraction-to-bounds)
            solver::ineq_soft::calculate_line_search_bounds(d, &setting_per_thread[tid]);
        });
        detail_timed_block_end("finalize_newton_step");
        detail_timed_block_start("corrector_step");
        finalize_bound_and_set_to_max();
        if (has_ineq && settings.ipm_enable_affine_step()) {
            // line search with max bounds
            graph_.for_each_parallel([&setting_per_thread](size_t tid, data *d) {
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
            graph_.for_each_parallel(solver::ineq_soft::first_order_correction_start);
            /// @todo compute the residuals
            // solve the problem again with updated mu
            detail_timed_block_start("riccati_recursion_correction");
            graph_.apply_backward(solver::ns_riccati::riccati_recursion_correction, true);
            detail_timed_block_end("riccati_recursion_correction");
            graph_.for_each_parallel(solver::ns_riccati::compute_primal_sensitivity_correction);
            graph_.apply_forward(solver::ns_riccati::fwd_linear_rollout_correction, true);
            graph_.for_each_parallel([n_iter](data *d) {
                solver::ineq_soft::first_order_correction_end(d);
                solver::ns_riccati::finalize_newton_step_correction(d);
                solver::ineq_soft::finalize_newton_step(d, false);
                solver::ns_riccati::finalize_dual_newton_step(d);
            });
            // recompute line search bounds with the corrected newton step
            settings.ls_config_reset();
            for (size_t i : range(n_worker)) {
                setting_per_thread[i].ls_config_reset();
            }
            graph_.for_each_parallel([&setting_per_thread](size_t tid, data *d) {
                solver::ineq_soft::calculate_line_search_bounds(d, &setting_per_thread[tid]);
            });
            finalize_bound_and_set_to_max();
        }
        detail_timed_block_end("corrector_step");
        // iterative refinement if the step is too small
        if (has_ineq) {
            kkt_info info;
            for (auto n : graph_.flatten_nodes()) {
                for (auto f : primal_fields)
                    info.inf_prim_step = std::max(info.inf_prim_step, n->prim_step[__x].cwiseAbs().maxCoeff());
                for (auto f : constr_fields) {
                    if (n->dual_step[f].size() > 0)
                        info.inf_dual_step = std::max(info.inf_dual_step, n->dual_step[f].cwiseAbs().maxCoeff());
                }
            }
            // if (info.inf_prim_step < 1e-1 || info.inf_dual_step < 1e-1) {
            size_t iter_refine_max = 5;
            size_t iter_refine = 0;
            detail_timed_block_start("iterative_refinement");
            while (iter_refine < iter_refine_max) {
                detail_timed_block_start("check_residual");
                graph_.for_each_parallel(solver::ns_riccati::compute_kkt_residual);
                detail_timed_block_end("check_residual");
                struct MOTO_ALIGN_NO_SHARING inf_res_state_worker {
                    scalar_t inf_res_stat_u = 0.;
                    scalar_t inf_res_stat_y = 0.;
                } thread_res[n_worker];
                size_t step = 0;
                graph_.apply_forward<true>([&](size_t tid, data *d, data *next) {
                    thread_res[tid].inf_res_stat_u = std::max(thread_res[tid].inf_res_stat_u, d->dense().res_stat_[__u].cwiseAbs().maxCoeff());
                    next->dense().res_stat_[__x].applyOnTheRight(utils::permutation_from_y_to_x(&d->problem(), &next->problem()));
                    d->dense().res_stat_[__y] += next->dense().res_stat_[__x];
                    thread_res[tid].inf_res_stat_y = std::max(thread_res[tid].inf_res_stat_y, d->dense().res_stat_[__y].cwiseAbs().maxCoeff());
                });
                // fmt::print("  iterative refinement {}, res_stat_u: {:.3e}, res_stat_y: {:.3e}\n",
                //            iter_refine, inf_res_stat_u, inf_res_stat_y);
                scalar_t inf_res_stat_u = 0.;
                scalar_t inf_res_stat_y = 0.;
                for (auto &w : thread_res) {
                    inf_res_stat_u = std::max(inf_res_stat_u, w.inf_res_stat_u);
                    inf_res_stat_y = std::max(inf_res_stat_y, w.inf_res_stat_y);
                }
                if (inf_res_stat_u < 1e-10 && inf_res_stat_y < 1e-10) {
                    break;
                }
                detail_timed_block_start("iterative_refinement_step");
                detail_timed_block_start("iterative_refinement_presolve");
                graph_.for_each_parallel(iterative_refinement_start);
                detail_timed_block_end("iterative_refinement_presolve");
                detail_timed_block_start("riccati_recursion_correction");
                graph_.apply_backward(solver::ns_riccati::riccati_recursion_correction, true);
                detail_timed_block_end("riccati_recursion_correction");
                graph_.for_each_parallel(solver::ns_riccati::compute_primal_sensitivity_correction);
                graph_.apply_forward(solver::ns_riccati::fwd_linear_rollout_correction, true);
                detail_timed_block_start("iterative_refinement_step_finalize");
                graph_.for_each_parallel(iterative_refinement_end);
                detail_timed_block_end("iterative_refinement_step_finalize");
                detail_timed_block_end("iterative_refinement_step");
                // recompute line search bounds with the corrected newton step
                settings.ls_config_reset();
                for (size_t i : range(n_worker)) {
                    setting_per_thread[i].ls_config_reset();
                }
                graph_.for_each_parallel([&setting_per_thread](size_t tid, data *d) {
                    solver::ineq_soft::calculate_line_search_bounds(d, &setting_per_thread[tid]);
                });
                finalize_bound_and_set_to_max();
                iter_refine++;
            }
            detail_timed_block_end("iterative_refinement");
            // }
        }
        /// @todo: update the line search stepsize?
        // real line search step
        detail_timed_block_start("line_search_step");
        graph_.for_each_parallel([this](data *d) {
            solver::ns_riccati::apply_affine_step(d, &settings);
            solver::ineq_soft::apply_affine_step(d, &settings);
        });
        detail_timed_block_end("line_search_step");
        if (i_iter + 1 == n_iter) {
            fmt::print("after line search step\n");
            // graph_.apply_forward(solver::ns_riccati::compute_kkt_residual);
            // settings.mu_method = solver::ipm_config::quality_function_based;
        }
        detail_timed_block_start("update_approx");
        graph_.for_each_parallel(data::update_approx);
        detail_timed_block_end("update_approx");
        // if (i_iter + 1 == n_iter) {
        //     graph_.apply_forward([](data *d) {
        //         d->merge_jacobian_modification();
        //     });
        // }
        detail_timed_block_start("update_res_stat");
        kkt_info info = compute_kkt_info();
        detail_timed_block_end("update_res_stat");
        timed_block_end("sqp_single_iter");
        // print statistics
        print_stats(i_iter, info, has_ineq);
        // });
    }
}
ns_sqp::kkt_info ns_sqp::compute_kkt_info() {
    kkt_info info;
    for (auto n : graph_.flatten_nodes()) {
        info.objective += n->cost();
        info.inf_prim_res = std::max(info.inf_prim_res, n->inf_prim_res_);
        info.inf_dual_res = std::max(info.inf_dual_res, n->dense().jac_[__u].cwiseAbs().maxCoeff());
        info.inf_comp_res = std::max(info.inf_comp_res, n->inf_comp_res_);
        for (auto f : primal_fields)
            info.inf_prim_step = std::max(info.inf_prim_step, n->prim_step[__x].cwiseAbs().maxCoeff());
        for (auto f : constr_fields) {
            if (n->dual_step[f].size() > 0)
                info.inf_dual_step = std::max(info.inf_dual_step, n->dual_step[f].cwiseAbs().maxCoeff());
        }
    }
    size_t step = 0;
    graph_.apply_forward(
        [&step, &info](node_data *cur, node_data *next) {
            if (next != nullptr) [[likely]] {
                // cancellation of jacobian from y to x
                static row_vector tmp;
                tmp.conservativeResize(next->dense().jac_[__x].cols());
                tmp.noalias() = next->dense().jac_[__x] *
                                    utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                                cur->dense().jac_[__y];
                info.inf_dual_res = std::max(info.inf_dual_res, tmp.cwiseAbs().maxCoeff());
            } else /// @todo: include initial jac[__x] inf norm if init is optimized
                info.inf_dual_res = std::max(info.inf_dual_res, cur->dense().jac_[__y].cwiseAbs().maxCoeff());
            // fmt::println("------ step {} dual_res: ", step++);
            // fmt::println("x dual res {}", cur->dense().jac_[__x]);
            // fmt::println("y dual res {}", cur->dense().jac_[__y]);
            // fmt::println("u dual res {}", cur->dense().jac_[__u]);
            // // fmt::println("prim {}: {}", step, cur->prim_step[__x
            // // fmt::println("prim {}: {}", step, cur->value(__u).transpose());
            // for (auto f : constr_fields) {
            //     if (cur->dense().dual_[f].size() > 0)
            //         fmt::println("dual {}: {}", f, cur->dense().dual_[f].transpose());
            // }
            // fmt::println("dual {}: {}", step, cur->dense().dual_[__ineq_xu].transpose());
            // fmt::println("jac  {}: {}", step++, cur->dense().jac_[__u]);
            // fmt::println("{}", cur->dense().jac_[__y].cwiseAbs().maxCoeff());
        },
        true);
    return info;
}
} // namespace moto
