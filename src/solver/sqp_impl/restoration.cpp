#include <algorithm>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>

namespace moto {
namespace {

template <typename Fn>
void for_each_overlay_pair(ns_sqp::storage_type &outer_graph,
                           ns_sqp::storage_type &resto_graph,
                           Fn &&fn) {
    auto &outer_nodes = outer_graph.flatten_nodes();
    auto &resto_nodes = resto_graph.flatten_nodes();
    if (outer_nodes.size() != resto_nodes.size()) {
        throw std::runtime_error("restoration overlay graph/node mismatch");
    }
    parallel_for(0, outer_nodes.size(), [&](size_t i) {
        fn(*outer_nodes[i], *resto_nodes[i]);
    });
}

} // namespace

ns_sqp::result_type ns_sqp::restoration_update(const kkt_info &kkt_before, const iter_info &iter_before, filter_linesearch_data &ls) {
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }

    auto &outer_graph = active_data();
    auto &resto_graph = restoration_graph();
    if (settings.verbose) {
        fmt::print("\n=== enter restoration ===\n");
        fmt::print("  entry iter={}  outer aug_obj={:.3e}  outer ls_obj={:.3e}  prim={:.3e}  dual={:.3e}  comp={:.3e}\n",
                   iter_before.num_iter, kkt_before.barrier_objective.augmented_objective, kkt_before.barrier_objective.ls_objective,
                   kkt_before.primal.res_l1, kkt_before.dual.inf_res, kkt_before.primal.inf_comp);
    }
    settings.in_restoration = true;
    set_phase_graph_override(resto_graph);

    auto rest_state = result_type{
        .iter = iter_info{
            .num_iter = iter_before.num_iter,
        }};

    const auto refresh_restoration_derivatives = [&]() {
        resto_graph.for_each_parallel([this](data *d) {
            d->update_approximation(node_data::update_mode::eval_derivatives, true);
        });
    };
    const auto initialize_restoration_problem = [&]() {
        const auto prox_eps = scalar_t(1.0);
        for_each_overlay_pair(outer_graph, resto_graph, [&](data &outer, data &resto) {
            solver::restoration::sync_outer_to_restoration_state(outer, resto, prox_eps, &settings.mu);
        });
        resto_graph.for_each_parallel([this](data *d) {
            d->for_each_constr([this](const generic_func &c, func_approx_data &fd) {
                c.setup_workspace_data(fd, &settings);
            });
            solver::ineq_soft::bind_and_invalidate(d);
            d->update_approximation(node_data::update_mode::eval_val, true);
        });
        resto_graph.for_each_parallel([](data *d) {
            d->update_approximation(node_data::update_mode::eval_all, true);
        });
    };
    const auto evaluate_outer_trial_from_restoration = [&]() {
        // Bounce through the outer graph only long enough to evaluate the
        // candidate in normal-phase metrics, then restore the restoration phase.
        resto_graph.for_each_parallel([this](data *d) {
            d->update_approximation(node_data::update_mode::eval_val, true);
        });
        outer_graph.for_each_parallel([](data *d) { d->backup_trial_state(); });
        for_each_overlay_pair(resto_graph, outer_graph, [&](data &resto, data &outer) {
            solver::restoration::sync_restoration_to_outer_state(resto, outer);
        });
        outer_graph.for_each_parallel([this](data *d) {
            d->update_approximation(node_data::update_mode::eval_val, true);
        });
        clear_phase_graph_override();
        kkt_info outer_trial;
        /// no need update step info because the reference kkt is from outside (Backup)
        update_primal_info(outer_trial, point_value_mask::primal | point_value_mask::barrier_objective);
        set_phase_graph_override(resto_graph);
        outer_graph.for_each_parallel([](data *d) { d->restore_trial_state(); });
        refresh_restoration_derivatives();
        return outer_trial;
    };
    const auto finish_restoration = [&](bool success) {
        if (success) {
            for_each_overlay_pair(resto_graph, outer_graph, [&](data &resto, data &outer) {
                solver::restoration::sync_restoration_to_outer_state(resto, outer);
            });
            // apply fraction-to-boundary to the multipliers
            reset_ls_workers();
            outer_graph.for_each_parallel([this](size_t tid, data *d) {
                solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
            });
            solver::linesearch_config ls_dual_only;
            for (solver::linesearch_config &s : setting_per_thread) {
                ls_dual_only.dual.merge_from(s.dual);
            }
            // check multiplier reset threshold
            std::vector<bool> local_exceed_bound(setting_per_thread.size(), false);
            outer_graph.for_each_parallel([&, this](size_t tid, data *d) {
                d->for_each<ineq_constr_fields>([&](const ineq_constr &c, ineq_constr::data_map_t &id) {
                    c.restoration_commit_dual_step(id, ls_dual_only.dual.alpha_max);
                });
                for (auto field : ineq_constr_fields) {
                    local_exceed_bound[tid] =
                        local_exceed_bound[tid] ||
                        d->dense().dual_[field].cwiseAbs().maxCoeff() >
                            settings.restoration.bound_mult_reset_threshold;
                }
            });
            bool reset_bound_multipliers =
                std::any_of(local_exceed_bound.begin(), local_exceed_bound.end(), [](bool b) { return b; });

            if (settings.verbose) {
                fmt::println("[resto cleanup] reset_bound_multipliers: {}", reset_bound_multipliers);
            }

            clear_phase_graph_override();
            outer_graph.for_each_parallel([&, this](data *d) {
                d->update_approximation(node_data::update_mode::eval_all, true);
                if (reset_bound_multipliers) {
                    d->for_each<ineq_constr_fields>([&](const ineq_constr &c, ineq_constr::data_map_t &id) {
                        c.restoration_reset_bound_multipliers(id);
                    });
                }
            });

            if (settings.eq_init.enabled && settings.eq_init.rebuild_after_restoration_exit) {
                initialize_equality_multipliers();
            }
            // update the result with the final solution
            update_primal_info(rest_state, point_value_mask::primal | point_value_mask::barrier_objective);
            update_stat_info(rest_state);
        } else {
            clear_phase_graph_override();
        }
        settings.in_restoration = false;
    };

    initialize_restoration_problem();

    ls.augment_filter_for_restoration_start(kkt_before, settings);
    filter_linesearch_data rls;
    // Reset the switching threshold for the restoration subproblem. Reusing the
    // outer line-search threshold makes restoration switch to Armijo mode based
    // on the very first outer residual, which can be orders of magnitude looser
    // than the residual at restoration entry.
    rls.constr_vio_min =
        std::max(kkt_before.primal.res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);

    update_primal_info(rest_state, point_value_mask::primal | point_value_mask::barrier_objective);
    update_stat_info(rest_state);
    kkt_info kkt_outer_trial{};
    const size_t max_resto_iters =
        std::min(settings.restoration.max_iter,
                 settings.max_iter > iter_before.num_iter ? settings.max_iter - iter_before.num_iter : size_t(0));
    const scalar_t accepted_outer_prim_res =
        settings.restoration.restoration_improvement_frac * kkt_before.primal.res_l1;

    for (size_t i_rest = 0; i_rest < max_resto_iters; ++i_rest) {
        const line_search_action action = sqp_iter(rls, rest_state,
                                                   /*do_scaling=*/false,
                                                   /*do_refinement=*/true,
                                                   /*gauss_newton=*/false);
        rest_state.iter = iter_info{
            .num_iter = iter_before.num_iter + i_rest + 1,
        };
        if (settings.verbose) {
            print_stats(rest_state, rest_state.iter, rls.step_cnt);
        }

        bool resto_converged = rest_state.dual.inf_res < settings.dual_tol;

        if (action == line_search_action::accept) {
            kkt_outer_trial = evaluate_outer_trial_from_restoration();
            rest_state.iter = iter_info{
                .num_iter = iter_before.num_iter + i_rest + 1,
            };

            const bool outer_accept = outer_filter_accepts(ls, kkt_outer_trial, kkt_before);
            const bool prim_improved = kkt_outer_trial.primal.res_l1 < accepted_outer_prim_res;
            if (outer_accept && prim_improved) {
                rest_state.iter.result = iter_result_t::success;
                break;
            } else if (resto_converged) {
                rest_state.iter.result = iter_result_t::infeasible_stationary;
                break;
            }
            continue;
        }

        if (action == line_search_action::failure) {
            rest_state.iter.result = iter_result_t::restoration_failed;
            break;
        }
    }
    if (rest_state.iter.result == iter_result_t::unknown) {
        /// @note might be wrong
        rest_state.iter.result = iter_result_t::restoration_reached_max_iter;
    }
    finish_restoration(rest_state.iter.result == iter_result_t::success);
    if (settings.verbose) {
        fmt::print("[resto]: primal residual(L1): {} before {}\n", kkt_outer_trial.primal.res_l1, kkt_before.primal.res_l1);
        fmt::print("[resto]: primal residual(Linf): {} before {}\n", kkt_outer_trial.primal.inf_res, kkt_before.primal.inf_res);
        fmt::print("=== leave restoration: {} ===\n\n", magic_enum::enum_name<iter_result_t>(rest_state.iter.result));
    }
    return rest_state;
}

} // namespace moto
