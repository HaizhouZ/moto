#include <algorithm>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>

namespace moto {
namespace {

struct ipm_pair_snapshot {
    ineq_constr::approx_data::box_pair_runtime *pair;
    ineq_constr::approx_data::box_pair_runtime state;
};

std::vector<ipm_pair_snapshot> backup_outer_ipm_pairs(ns_sqp::storage_type &outer_graph) {
    std::vector<ipm_pair_snapshot> backup;
    solver::for_each(solver::seq, outer_graph, [&](node_data *node) {
        node->for_each<ineq_constr_fields>([&](const ineq_constr &, ineq_constr::data_map_t &id) {
            auto *ipm_data = dynamic_cast<solver::ipm_constr::ipm_data *>(&id);
            if (ipm_data == nullptr || !ipm_data->boxed()) {
                return;
            }
            const auto &box = ipm_data->require_box_spec("backup_outer_ipm_pairs");
            for (auto side : box_sides) if (box.has_side[side]) {
                auto &pair = *ipm_data->box_side_[side];
                backup.push_back({&pair, pair});
            }
        });
    });
    return backup;
}

void restore_outer_ipm_pairs(const std::vector<ipm_pair_snapshot> &backup) {
    for (const auto &state : backup) {
        *state.pair = state.state;
    }
}

void reset_restoration_bound_multipliers(node_data *node) {
    node->for_each<ineq_constr_fields>([](const ineq_constr &c, ineq_constr::data_map_t &id) {
        c.restoration_reset_bound_multipliers(id);
    });
}

void commit_restoration_dual_step(node_data *node, scalar_t alpha_dual) {
    node->for_each<ineq_constr_fields>([&](const ineq_constr &c, ineq_constr::data_map_t &id) {
        c.restoration_commit_dual_step(id, alpha_dual);
    });
}

bool exceeds_restoration_dual_bound(const node_data *node, scalar_t threshold) {
    for (auto field : ineq_constr_fields) {
        const auto &dual = node->dense().dual_[field];
        if (dual.size() > 0 && dual.cwiseAbs().maxCoeff() > threshold) {
            return true;
        }
    }
    return false;
}

} // namespace

ns_sqp::result_type ns_sqp::restoration_update(const kkt_info &kkt_before, const iter_info &iter_before,
                                               filter_linesearch_data &ls, size_t update_iter_limit) {
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }

    auto &outer_graph = active_data();
    auto &resto_graph = restoration_graph();
    const auto outer_ipm_backup = settings.has_ipm_ineq
                                      ? backup_outer_ipm_pairs(outer_graph)
                                      : std::vector<ipm_pair_snapshot>{};
    if (settings.verbose) {
        fmt::print("\n=== enter restoration ===\n");
        fmt::print("  entry iter={}  outer aug_obj={:.3e}  outer ls_obj={:.3e}  prim={:.3e}  dual={:.3e}  comp={:.3e}\n",
                   iter_before.num_iter, kkt_before.barrier_objective.augmented_objective, kkt_before.barrier_objective.ls_objective,
                   kkt_before.primal.res_l1, kkt_before.dual.inf_res, kkt_before.primal.inf_comp);
    }
    scoped_phase_graph_override phase_graph(*this, resto_graph, true);

    auto rest_state = result_type{
        .iter = iter_info{
            .num_iter = iter_before.num_iter,
        }};

    const auto initialize_restoration_problem = [&]() {
        const auto prox_eps = scalar_t(1.0);
        solver::for_each(solver::par, solver::zip(outer_graph, resto_graph),
                     [&](data *outer, data *resto) {
            solver::restoration::sync_outer_to_restoration_state(*outer, *resto, prox_eps, &settings.mu);
            resto->for_each_constr([this](const generic_constr &c, func_approx_data &fd) {
                c.setup_workspace_data(fd, &settings);
            });
            solver::ineq_soft::bind_and_invalidate(resto);
            resto->update_approximation(node_data::update_mode::eval_val, true);
        });
        solver::for_each(solver::par, resto_graph, [](data *d) {
            d->update_approximation(node_data::update_mode::eval_all, true);
        });
    };
    const auto evaluate_outer_trial_from_restoration = [&]() {
        // Bounce through the outer graph only long enough to evaluate the
        // candidate in normal-phase metrics, then restore the restoration phase.
        solver::for_each(solver::par, solver::zip(resto_graph, outer_graph),
                     [&](data *resto, data *outer) {
            outer->backup_primal_state();
            solver::ineq_soft::backup_trial_state(outer);
            solver::restoration::sync_restoration_candidate_to_outer_state(*resto, *outer);
            outer->update_approximation(node_data::update_mode::eval_val, true);
        });
        phase_graph.use_default_graph(false);
        kkt_info outer_trial;
        /// no need update step info because the reference kkt is from outside (Backup)
        update_primal_info(outer_trial, point_value_mask::primal | point_value_mask::barrier_objective);
        phase_graph.use_graph(resto_graph, true);
        solver::for_each(solver::par, outer_graph, [](data *d) {
            d->restore_primal_state();
            solver::ineq_soft::restore_trial_state(d);
        });
        return outer_trial;
    };

    const auto reset_outer_bound_multipliers = [&]() {
        solver::for_each(solver::par, outer_graph, reset_restoration_bound_multipliers);
    };

    const auto commit_restoration_state_and_get_dual_alpha = [&]() {
        reset_ls_workers();
        solver::for_each(solver::par, solver::zip(resto_graph, outer_graph),
                         [&](size_t tid, data *resto, data *outer) {
                             solver::restoration::commit_restoration_to_outer_state(*resto, *outer);
                             if (settings.has_ineq_soft) {
                                 solver::ineq_soft::update_ls_bounds(outer, &setting_per_thread[tid]);
                             }
                         });
        solver::linesearch_config ls_dual_only;
        for (solver::linesearch_config &s : setting_per_thread) {
            ls_dual_only.dual.merge_from(s.dual);
        }
        return ls_dual_only.dual.alpha_max;
    };

    const auto commit_restoration_dual_step_and_check_reset = [&](scalar_t alpha_dual) {
        std::vector<uint8_t> local_exceed_bound(setting_per_thread.size(), uint8_t{0});
        solver::for_each(solver::par, outer_graph, [&, this](size_t tid, data *d) {
            commit_restoration_dual_step(d, alpha_dual);
            local_exceed_bound[tid] = static_cast<uint8_t>(
                exceeds_restoration_dual_bound(d, settings.restoration.bound_mult_reset_threshold));
        });
        return std::any_of(local_exceed_bound.begin(), local_exceed_bound.end(), [](uint8_t b) { return b != 0; });
    };

    const auto update_outer_after_restoration = [&](bool reset_bound_multipliers, bool rebuild_eq_duals) {
        solver::for_each(solver::par, outer_graph, [=](data *d) {
            if (reset_bound_multipliers && !rebuild_eq_duals) {
                reset_restoration_bound_multipliers(d);
            }
            d->update_approximation(node_data::update_mode::eval_all, true);
        });
    };

    const auto finish_successful_restoration = [&]() {
        const bool rebuild_eq_duals =
            settings.eq_init.enabled && settings.eq_init.rebuild_after_restoration_exit;
        const scalar_t alpha_dual = commit_restoration_state_and_get_dual_alpha();
        const bool reset_bound_multipliers = commit_restoration_dual_step_and_check_reset(alpha_dual);

        if (settings.verbose) {
            fmt::println("[resto cleanup] reset_bound_multipliers: {}", reset_bound_multipliers);
        }

        phase_graph.use_default_graph(false);
        if (reset_bound_multipliers && rebuild_eq_duals) {
            reset_outer_bound_multipliers();
        }

        if (rebuild_eq_duals) {
            initialize_equality_multipliers(outer_graph, false);
        }
        update_outer_after_restoration(reset_bound_multipliers, rebuild_eq_duals);
        update_primal_info(rest_state, point_value_mask::primal | point_value_mask::barrier_objective);
        update_stat_info(rest_state);
    };

    const auto restore_failed_restoration = [&]() {
        restore_outer_ipm_pairs(outer_ipm_backup);
        phase_graph.use_default_graph(false);
        solver::for_each(solver::par, outer_graph, [](data *d) {
            d->update_approximation(node_data::update_mode::eval_val, true);
        });
    };

    const auto finish_restoration = [&](bool success) {
        if (success) {
            finish_successful_restoration();
        } else {
            restore_failed_restoration();
        }
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
                 update_iter_limit > iter_before.num_iter ? update_iter_limit - iter_before.num_iter : size_t(0));
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
