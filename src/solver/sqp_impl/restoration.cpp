#include <algorithm>
#include <cstdlib>
#include <limits>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
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

void dump_overlay_pair_order(ns_sqp::storage_type &outer_graph,
                             ns_sqp::storage_type &resto_graph,
                             size_t max_pairs = std::numeric_limits<size_t>::max()) {
    auto &outer_nodes = outer_graph.flatten_nodes();
    auto &resto_nodes = resto_graph.flatten_nodes();
    fmt::print("  [resto-pairs] first {} pairs:\n", std::min(max_pairs, outer_nodes.size()));
    for (size_t i = 0; i < std::min({max_pairs, outer_nodes.size(), resto_nodes.size()}); ++i) {
        fmt::print("    pair {}: outer_uid={} resto_uid={}\n",
                   i, outer_nodes[i]->problem().uid(), resto_nodes[i]->problem().uid());
    }
}

bool resto_eq_debug_enabled() {
    const char *flag = std::getenv("MOTO_RESTO_EQ_DEBUG");
    return flag != nullptr && std::string_view(flag) != "0";
}

bool resto_entry_debug_enabled() {
    const char *flag = std::getenv("MOTO_RESTO_ENTRY_DEBUG");
    return flag != nullptr && std::string_view(flag) != "0";
}

void dump_outer_overlay_source_values(ns_sqp::storage_type &outer_graph,
                                      ns_sqp::storage_type &resto_graph) {
    for_each_overlay_pair(outer_graph, resto_graph, [&](node_data &outer, node_data &resto) {
        auto dump_field = [&](field_t field) {
            resto.for_each(field, [&](const solver::restoration::resto_eq_elastic_constr &overlay,
                                      solver::restoration::resto_eq_elastic_constr::approx_data &) {
                const auto src_field = overlay.source()->field();
                const auto &outer_exprs = outer.problem().exprs(src_field);
                const size_t src_pos = outer.problem().pos(overlay.source());
                const auto &outer_expr = outer_exprs[src_pos];
                const func outer_func = outer_expr.cast<generic_func>();
                const auto &outer_ad = outer.data(outer_func);
                const auto &source = overlay.source();
                const auto &source_gf = source.as<generic_func>();
                const auto &outer_gf = outer_func.as<generic_func>();
                fmt::print("  [resto-source:{}] outer_node={} outer_prob={} src_field={} src_pos={} source_uid={} outer_uid={} same_uid={}\n",
                           overlay.name(),
                           outer.problem().uid(),
                           outer.problem().uid(),
                           field::name(src_field),
                           src_pos,
                           source->uid(),
                           outer_expr->uid(),
                           source->uid() == outer_expr->uid() ? "yes" : "no");
                fmt::print("    source_name={} outer_name={} outer_v=[",
                           source->name(), outer_expr->name());
                for (Eigen::Index i = 0; i < outer_ad.v_.size(); ++i) {
                    fmt::print("{}{:.6e}", i == 0 ? "" : ", ", outer_ad.v_(i));
                }
                fmt::print("]\n");
                const scalar_t y_diff =
                    (outer.sym_val().value_[__y].size() == resto.sym_val().value_[__y].size() && outer.sym_val().value_[__y].size() > 0)
                        ? (outer.sym_val().value_[__y] - resto.sym_val().value_[__y]).cwiseAbs().maxCoeff()
                        : scalar_t(0.);
                const scalar_t p_diff =
                    (outer.sym_val().value_[__p].size() == resto.sym_val().value_[__p].size() && outer.sym_val().value_[__p].size() > 0)
                        ? (outer.sym_val().value_[__p] - resto.sym_val().value_[__p]).cwiseAbs().maxCoeff()
                        : scalar_t(0.);
                fmt::print("    sync_diff: |y_outer-y_resto|_inf={:.3e} |p_outer-p_resto|_inf={:.3e}\n", y_diff, p_diff);
                if (!source_gf.in_args().empty() && !outer_gf.in_args().empty()) {
                    fmt::print("    source_first_arg_field={} outer_first_arg_field={} source_nargs={} outer_nargs={}\n",
                               field::name(source_gf.in_args().front()->field()),
                               field::name(outer_gf.in_args().front()->field()),
                               source_gf.in_args().size(),
                               outer_gf.in_args().size());
                    fmt::print("    source_arg_sig=[");
                    for (size_t i = 0; i < source_gf.in_args().size(); ++i) {
                        const auto &arg = source_gf.in_args()[i];
                        fmt::print("{}{}:{}", i == 0 ? "" : ", ", field::name(arg->field()), arg->dim());
                    }
                    fmt::print("] outer_arg_sig=[");
                    for (size_t i = 0; i < outer_gf.in_args().size(); ++i) {
                        const auto &arg = outer_gf.in_args()[i];
                        fmt::print("{}{}:{}", i == 0 ? "" : ", ", field::name(arg->field()), arg->dim());
                    }
                    fmt::print("]\n");
                }
            });
        };
        dump_field(__eq_x_soft);
        dump_field(__eq_xu_soft);
    });
}

void dump_outer_entry_equalities(ns_sqp::storage_type &graph) {
    for (auto *n : graph.flatten_nodes()) {
        auto dump_field = [&](field_t field) {
            n->for_each(field, [&](const generic_constr &c, func_approx_data &ad) {
                if (c.name() != "arm_ee_constr") {
                    return;
                }
                fmt::print("  [resto-entry:{}:{}] v=[", n->problem().uid(), c.name());
                for (Eigen::Index i = 0; i < ad.v_.size(); ++i) {
                    fmt::print("{}{:.6e}", i == 0 ? "" : ", ", ad.v_(i));
                }
                fmt::print("]\n");
            });
        };
        dump_field(__eq_x);
        dump_field(__eq_xu);
    }
}

void dump_outer_entry_inequalities(ns_sqp::storage_type &graph) {
    bool printed = false;
    scalar_t worst_tq_v = 0.;
    size_t worst_tq_uid = 0;
    Eigen::Index worst_tq_idx = -1;
    for (auto *n : graph.flatten_nodes()) {
        auto dump_field = [&](field_t field) {
            n->for_each(field, [&](const solver::ipm_constr &c, solver::ipm_constr::ipm_data &ipm_d) {
                Eigen::Index local_idx = -1;
                const scalar_t local_worst = ipm_d.v_.size() ? ipm_d.slack_.cwiseAbs().minCoeff(&local_idx) : 0.;
                if (local_worst > worst_tq_v) {
                    worst_tq_v = local_worst;
                    worst_tq_uid = n->problem().uid();
                    worst_tq_idx = local_idx;
                }
                if (printed) {
                    return;
                }
                fmt::print("  [resto-entry:{}:{}] g=[", n->problem().uid(), c.name());
                for (Eigen::Index i = 0; i < ipm_d.g_.size(); ++i) {
                    fmt::print("{}{:.6e}", i == 0 ? "" : ", ", ipm_d.g_(i));
                }
                fmt::print("]\n");
                fmt::print("  [resto-entry:{}:{}] slack=[", n->problem().uid(), c.name());
                for (Eigen::Index i = 0; i < ipm_d.slack_.size(); ++i) {
                    fmt::print("{}{:.6e}", i == 0 ? "" : ", ", ipm_d.slack_(i));
                }
                fmt::print("]\n");
                fmt::print("  [resto-entry:{}:{}] v=[", n->problem().uid(), c.name());
                for (Eigen::Index i = 0; i < ipm_d.v_.size(); ++i) {
                    fmt::print("{}{:.6e}", i == 0 ? "" : ", ", ipm_d.v_(i));
                }
                fmt::print("]\n");
                // printed = true;
            });
        };
        dump_field(__ineq_x);
        dump_field(__ineq_xu);
    }
    fmt::print("  worst slack {} at node {} idx {}", worst_tq_v, worst_tq_uid, worst_tq_idx);
}

void dump_resto_eq_node(ns_sqp::data &d, std::string_view label) {
    fmt::print("=== resto-eq-debug:{} node={} ===\n", label, d.problem().uid());
    fmt::print("  lag_jac_corr[x]={:.3e} lag_jac_corr[u]={:.3e} lag_jac_corr[y]={:.3e}\n",
               d.dense().lag_jac_corr_[__x].size() ? d.dense().lag_jac_corr_[__x].cwiseAbs().maxCoeff() : 0.,
               d.dense().lag_jac_corr_[__u].size() ? d.dense().lag_jac_corr_[__u].cwiseAbs().maxCoeff() : 0.,
               d.dense().lag_jac_corr_[__y].size() ? d.dense().lag_jac_corr_[__y].cwiseAbs().maxCoeff() : 0.);
    fmt::print("  Q_yx_mod={:.3e} finite={}  Q_yy_mod={:.3e} finite={}\n",
               d.Q_yx_mod.dense().size() ? d.Q_yx_mod.dense().cwiseAbs().maxCoeff() : 0.,
               d.Q_yx_mod.dense().allFinite(),
               d.Q_yy_mod.dense().size() ? d.Q_yy_mod.dense().cwiseAbs().maxCoeff() : 0.,
               d.Q_yy_mod.dense().allFinite());
    auto print_overlay_summary = [&](field_t field) {
        d.for_each(field, [&](const solver::restoration::resto_eq_elastic_constr &overlay,
                              solver::restoration::resto_eq_elastic_constr::approx_data &ad) {
            const auto residuals =
                solver::restoration::resto_eq_elastic_constr::current_local_residuals(ad.elastic);
            const size_t src_pos = d.problem().pos(overlay.source());
            fmt::print("  {} src_pos={} |v|={:.3e} |base|={:.3e} schur_inv={:.3e} stat={:.3e} comp={:.3e}\n",
                       overlay.name(), src_pos,
                       ad.v_.size() ? ad.v_.cwiseAbs().maxCoeff() : 0.,
                       ad.base_residual.size() ? ad.base_residual.cwiseAbs().maxCoeff() : 0.,
                       ad.elastic.schur_inv_diag.size() ? ad.elastic.schur_inv_diag.cwiseAbs().maxCoeff() : 0.,
                       residuals.inf_stat, residuals.inf_comp);
        });
    };
    print_overlay_summary(__eq_x_soft);
    print_overlay_summary(__eq_xu_soft);
}

} // namespace

ns_sqp::kkt_info ns_sqp::restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls) {
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }

    auto &outer_graph = active_data();
    auto &resto_graph = restoration_graph();
    if (settings.verbose) {
        fmt::print("\n=== enter restoration ===\n");
        fmt::print("  entry iter={}  outer objective={:.3e}  outer search_obj={:.3e}  prim={:.3e}  dual={:.3e}  comp={:.3e}\n",
                   kkt_before.num_iter, kkt_before.objective, kkt_before.penalized_obj,
                   kkt_before.prim_res_l1, kkt_before.inf_dual_res, kkt_before.inf_comp_res);
        if (resto_entry_debug_enabled()) {
            dump_outer_entry_equalities(outer_graph);
            dump_outer_entry_inequalities(outer_graph);
        }
    }
    settings.in_restoration = true;
    set_phase_graph_override(resto_graph);
    const auto compute_restoration_info_from_split = [&](bool update_dual_res) {
        kkt_info kkt;
        kkt = compute_prim_res_info(kkt);
        kkt = compute_ls_info(kkt);
        if (update_dual_res) {
            kkt = compute_dual_res_info(kkt);
        }
        return kkt;
    };
    const auto compute_normal_info_from_split = [&](bool update_dual_res, bool include_ls_info) {
        kkt_info kkt;
        kkt = compute_prim_res_info(kkt);
        if (include_ls_info) {
            kkt = compute_ls_info(kkt);
        }
        if (update_dual_res) {
            kkt = compute_dual_res_info(kkt);
        }
        return kkt;
    };

    const auto set_iter_meta = [&](kkt_info &info, size_t iter_offset, size_t ls_steps) {
        info.num_iter = kkt_before.num_iter + iter_offset;
        info.ls_steps = ls_steps;
    };
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
        if (settings.verbose && resto_entry_debug_enabled()) {
            dump_overlay_pair_order(outer_graph, resto_graph);
            dump_outer_overlay_source_values(outer_graph, resto_graph);
        }
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
        if (resto_eq_debug_enabled()) {
            auto nodes = resto_graph.flatten_nodes();
            if (!nodes.empty()) {
                dump_resto_eq_node(*nodes.back(), "after-derivatives");
            }
        }
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
        outer_trial = compute_prim_res_info(outer_trial);
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

            reset_ls_workers();
            outer_graph.for_each_parallel([this](size_t tid, data *d) {
                solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
            });
            solver::linesearch_config ls_dual_only;
            for (solver::linesearch_config &s : setting_per_thread) {
                ls_dual_only.dual.merge_from(s.dual);
            }

            std::vector<bool> local_exceed_bound(setting_per_thread.size(), false);
            outer_graph.for_each_parallel([&, this](size_t tid, data *d) {
                solver::ineq_soft::for_each(
                    d, [&](const soft_constr &sf, soft_constr::data_map_t &sd) {
                        if (in_field(sf.field(), ineq_constr_fields)) {
                            sd.multiplier_ += ls_dual_only.dual.alpha_max * sd.d_multiplier_;
                        }
                    });
                for (auto field : ineq_constr_fields) {
                    local_exceed_bound[tid] =
                        local_exceed_bound[tid] ||
                        d->dense().dual_[field].cwiseAbs().maxCoeff() >
                            settings.restoration.bound_mult_reset_threshold;
                }
            });

            const bool reset_bound_multipliers =
                std::any_of(local_exceed_bound.begin(), local_exceed_bound.end(), [](bool b) { return b; });
            if (settings.verbose) {
                fmt::println("[resto cleanup] reset_bound_multipliers: {}", reset_bound_multipliers);
            }

            clear_phase_graph_override();
            outer_graph.for_each_parallel([&, this](data *d) {
                d->update_approximation(node_data::update_mode::eval_val, true);
                if (reset_bound_multipliers) {
                    for (auto field : ineq_constr_fields) {
                        d->dense().dual_[field].setConstant(1.0);
                    }
                }
            });

            if (settings.eq_init.enabled && settings.eq_init.rebuild_after_restoration_exit) {
                initialize_equality_multipliers();
            } else {
                outer_graph.for_each_parallel([this](data *d) {
                    d->update_approximation(node_data::update_mode::eval_derivatives, true);
                });
            }
        } else {
            clear_phase_graph_override();
            outer_graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_all, true);
            });
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
        std::max(kkt_before.prim_res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);

    kkt_info kkt_rest = compute_restoration_info_from_split(true);
    set_iter_meta(kkt_rest, 0, 0);

    kkt_info kkt_outer_trial{};
    bool resto_accept = false;
    bool resto_stalled = false;
    bool resto_converged = false;
    const size_t max_resto_iters =
        std::min(settings.restoration.max_iter,
                 settings.max_iter > kkt_before.num_iter ? settings.max_iter - kkt_before.num_iter : size_t(0));
    const scalar_t accepted_outer_prim_res =
        settings.restoration.restoration_improvement_frac * kkt_before.prim_res_l1;

    for (size_t i_rest = 0; i_rest < max_resto_iters; ++i_rest) {
        const line_search_action action = sqp_iter(rls, kkt_rest,
                                                   /*do_scaling=*/false,
                                                   /*do_refinement=*/true,
                                                   /*gauss_newton=*/false);
        set_iter_meta(kkt_rest, i_rest + 1, rls.step_cnt);
        if (settings.verbose) {
            print_stats(kkt_rest);
        }

        resto_converged = kkt_rest.inf_dual_res < settings.dual_tol;

        if (action == line_search_action::accept) {
            kkt_outer_trial = evaluate_outer_trial_from_restoration();
            kkt_rest = compute_restoration_info_from_split(true);
            set_iter_meta(kkt_rest, i_rest + 1, rls.step_cnt);

            const bool outer_accept = outer_filter_accepts(ls, kkt_outer_trial, kkt_before);
            const bool prim_improved = kkt_outer_trial.prim_res_l1 < accepted_outer_prim_res;
            resto_accept = outer_accept && prim_improved;
            if (resto_accept || resto_converged) {
                break;
            }
            continue;
        }

        if (action == line_search_action::failure) {
            resto_graph.for_each_parallel([this](data *d) {
                d->restore_trial_state();
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
            resto_stalled = true;
            break;
        }
    }

    finish_restoration(resto_accept);
    kkt_info result = resto_accept ? compute_normal_info_from_split(true, false) : kkt_rest;
    result.num_iter = kkt_rest.num_iter;
    result.ls_steps = kkt_rest.ls_steps;
    if (resto_accept) {
        result.result = iter_result_t::unknown;
    } else if (resto_stalled) {
        result.result = iter_result_t::restoration_failed;
    } else if (resto_converged) {
        result.result = iter_result_t::infeasible_stationary;
    } else {
        result.result = iter_result_t::restoration_reached_max_iter;
    }
    if (settings.verbose) {
        fmt::print("[resto]: primal residual(L1): {} before {}\n", kkt_outer_trial.prim_res_l1, kkt_before.prim_res_l1);
        fmt::print("[resto]: primal residual(Linf): {} before {}\n", kkt_outer_trial.inf_prim_res, kkt_before.inf_prim_res);
        fmt::print("=== leave restoration: {} ===\n\n",
                   resto_accept ? "success" : (result.result == iter_result_t::infeasible_stationary ? "infeasible_stationary" : (result.result == iter_result_t::restoration_reached_max_iter ? "reached_max_iter" : "failed")));
    }
    return result;
}

} // namespace moto
