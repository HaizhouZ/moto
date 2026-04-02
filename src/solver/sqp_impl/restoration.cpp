#include <moto/solver/ns_sqp.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>
#include <cstdlib>

namespace moto {
namespace {

template <typename Fn>
void for_each_overlay_pair(ns_sqp::solver_graph_type &outer_graph,
                           ns_sqp::solver_graph_type &resto_graph,
                           Fn &&fn) {
    auto outer_nodes = outer_graph.flatten_nodes();
    auto resto_nodes = resto_graph.flatten_nodes();
    if (outer_nodes.size() != resto_nodes.size()) {
        throw std::runtime_error("restoration overlay graph/node mismatch");
    }
    for (size_t i = 0; i < outer_nodes.size(); ++i) {
        fn(*outer_nodes[i], *resto_nodes[i]);
    }
}

void sync_primal_state(node_data &src, node_data &dst) {
    for (auto field : primal_fields) {
        dst.sym_val().value_[field] = src.sym_val().value_[field];
    }
}

void sync_hard_duals(node_data &src, node_data &dst) {
    for (auto field : hard_constr_fields) {
        if (src.dense().dual_[field].size() == 0 || dst.dense().dual_[field].size() == 0) {
            continue;
        }
        dst.dense().dual_[field] = src.dense().dual_[field];
    }
}

bool resto_eq_debug_enabled() {
    const char *flag = std::getenv("MOTO_RESTO_EQ_DEBUG");
    return flag != nullptr && std::string_view(flag) != "0";
}

void dump_resto_eq_node(ns_sqp::data &d, std::string_view label) {
    fmt::print("=== resto-eq-debug:{} node={} ===\n", label, d.problem().uid());
    fmt::print("  dims: eq_x={} eq_xu={} eq_x_soft={} eq_xu_soft={} ineq_x={} ineq_xu={}\n",
               d.problem().dim(__eq_x), d.problem().dim(__eq_xu),
               d.problem().dim(__eq_x_soft), d.problem().dim(__eq_xu_soft),
               d.problem().dim(__ineq_x), d.problem().dim(__ineq_xu));
    fmt::print("  lag_jac_corr[x]={:.3e} lag_jac_corr[u]={:.3e} lag_jac_corr[y]={:.3e}\n",
               d.dense().lag_jac_corr_[__x].size() ? d.dense().lag_jac_corr_[__x].cwiseAbs().maxCoeff() : 0.,
               d.dense().lag_jac_corr_[__u].size() ? d.dense().lag_jac_corr_[__u].cwiseAbs().maxCoeff() : 0.,
               d.dense().lag_jac_corr_[__y].size() ? d.dense().lag_jac_corr_[__y].cwiseAbs().maxCoeff() : 0.);
    fmt::print("  Q_yx_mod={:.3e} finite={}  Q_yy_mod={:.3e} finite={}\n",
               d.Q_yx_mod.dense().size() ? d.Q_yx_mod.dense().cwiseAbs().maxCoeff() : 0.,
               d.Q_yx_mod.dense().allFinite(),
               d.Q_yy_mod.dense().size() ? d.Q_yy_mod.dense().cwiseAbs().maxCoeff() : 0.,
               d.Q_yy_mod.dense().allFinite());
    d.for_each(__eq_x_soft, [&](const solver::restoration::resto_eq_elastic_constr &overlay,
                                solver::restoration::resto_eq_elastic_constr::approx_data &ad) {
        fmt::print("  eq_soft {} src_field={} src_pos={}\n",
                   overlay.name(), static_cast<int>(overlay.source_field()), overlay.source_pos());
        fmt::print("    base={:.3e} v={:.3e} lambda={:.3e}\n",
                   ad.base_residual.size() ? ad.base_residual.cwiseAbs().maxCoeff() : 0.,
                   ad.v_.size() ? ad.v_.cwiseAbs().maxCoeff() : 0.,
                   ad.multiplier_.size() ? ad.multiplier_.cwiseAbs().maxCoeff() : 0.);
        fmt::print("    minv_bc={:.3e} minv_diag={:.3e} local_stat={:.3e} local_comp={:.3e}\n",
                   ad.elastic.minv_bc.size() ? ad.elastic.minv_bc.cwiseAbs().maxCoeff() : 0.,
                   ad.elastic.minv_diag.size() ? ad.elastic.minv_diag.cwiseAbs().maxCoeff() : 0.,
                   solver::restoration::current_local_residuals(ad.elastic).inf_stat,
                   solver::restoration::current_local_residuals(ad.elastic).inf_comp);
    });
    d.for_each(__eq_xu_soft, [&](const solver::restoration::resto_eq_elastic_constr &overlay,
                                 solver::restoration::resto_eq_elastic_constr::approx_data &ad) {
        fmt::print("  eq_xu_soft {} src_field={} src_pos={}\n",
                   overlay.name(), static_cast<int>(overlay.source_field()), overlay.source_pos());
        fmt::print("    base={:.3e} v={:.3e} lambda={:.3e}\n",
                   ad.base_residual.size() ? ad.base_residual.cwiseAbs().maxCoeff() : 0.,
                   ad.v_.size() ? ad.v_.cwiseAbs().maxCoeff() : 0.,
                   ad.multiplier_.size() ? ad.multiplier_.cwiseAbs().maxCoeff() : 0.);
        fmt::print("    minv_bc={:.3e} minv_diag={:.3e} local_stat={:.3e} local_comp={:.3e}\n",
                   ad.elastic.minv_bc.size() ? ad.elastic.minv_bc.cwiseAbs().maxCoeff() : 0.,
                   ad.elastic.minv_diag.size() ? ad.elastic.minv_diag.cwiseAbs().maxCoeff() : 0.,
                   solver::restoration::current_local_residuals(ad.elastic).inf_stat,
                   solver::restoration::current_local_residuals(ad.elastic).inf_comp);
    });
}

} // namespace

ns_sqp::kkt_info ns_sqp::restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls) {
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }

    auto &outer_graph = solver_graph();
    auto &resto_graph = restoration_graph();
    if (settings.verbose) {
        fmt::print("\n=== enter restoration ===\n");
        fmt::print("  entry iter={}  outer objective={:.3e}  outer search_obj={:.3e}  prim={:.3e}  dual={:.3e}  comp={:.3e}\n",
                   kkt_before.num_iter, kkt_before.objective, kkt_before.penalized_obj,
                   kkt_before.inf_prim_res, kkt_before.inf_dual_res, kkt_before.inf_comp_res);
    }
    settings.in_restoration = true;
    set_phase_graph_override(resto_graph);

    const auto prox_eps = scalar_t(1.0);
    for_each_overlay_pair(outer_graph, resto_graph, [&](data &outer, data &resto) {
        solver::restoration::sync_restoration_overlay_primal(outer, resto);
        solver::restoration::sync_restoration_overlay_duals(outer, resto);
        solver::restoration::seed_restoration_overlay_refs(resto, prox_eps);
    });

    resto_graph.for_each_parallel([this](data *d) {
        d->for_each_constr([this](const generic_func &c, func_approx_data &fd) { c.setup_workspace_data(fd, &settings); });
        d->update_approximation(node_data::update_mode::eval_val, true);
        solver::ineq_soft::initialize(d);
        d->update_approximation(node_data::update_mode::eval_derivatives, true);
    });
    if (resto_eq_debug_enabled()) {
        auto nodes = resto_graph.flatten_nodes();
        if (!nodes.empty()) {
            dump_resto_eq_node(*nodes.back(), "after-derivatives");
        }
    }

    ls.augment_filter_for_restoration_start(kkt_before, settings);
    filter_linesearch_data rls;
    rls.constr_vio_min = ls.constr_vio_min;

    kkt_info kkt_rest = compute_kkt_info();
    kkt_info kkt_outer_trial{};
    bool resto_accept = false;
    bool resto_stalled = false;
    size_t stalled_iters = 0;
    const size_t max_resto_iters =
        std::min(settings.restoration.max_iter,
                 settings.max_iter > kkt_before.num_iter ? settings.max_iter - kkt_before.num_iter : size_t(0));

    for (size_t i_rest = 0; i_rest < max_resto_iters; ++i_rest) {
        line_search_action action = sqp_iter(rls, kkt_rest,
                                             /*do_scaling=*/false,
                                             /*do_refinement=*/true,
                                             /*gauss_newton=*/false);
        kkt_rest.num_iter = kkt_before.num_iter + i_rest + 1;
        kkt_rest.ls_steps = rls.step_cnt;
        if (settings.verbose) {
            print_stats(kkt_rest);
        }

        if (action == line_search_action::accept) {
            resto_graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_val, true);
            });

            outer_graph.for_each_parallel([](data *d) { d->backup_trial_state(); });
            for_each_overlay_pair(resto_graph, outer_graph, [&](data &resto, data &outer) {
                sync_primal_state(resto, outer);
                sync_hard_duals(resto, outer);
            });
            outer_graph.for_each_parallel([](data *d) {
                d->update_approximation(node_data::update_mode::eval_val, true);
            });
            clear_phase_graph_override();
            kkt_outer_trial = compute_kkt_info_for_phase(iteration_phase::normal, false);
            set_phase_graph_override(resto_graph);
            outer_graph.for_each_parallel([](data *d) {
                d->restore_trial_state();
            });

            resto_graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
            kkt_rest = compute_kkt_info();
            kkt_rest.num_iter = kkt_before.num_iter + i_rest + 1;
            kkt_rest.ls_steps = rls.step_cnt;

            const bool outer_accept = outer_filter_accepts(ls, kkt_outer_trial, kkt_before);
            const bool prim_improved =
                kkt_outer_trial.inf_prim_res < settings.restoration.restoration_improvement_frac * kkt_before.inf_prim_res;
            if (outer_accept && prim_improved) {
                resto_accept = true;
                break;
            }
        }

        if (action == line_search_action::failure) {
            resto_graph.for_each_parallel([this](data *d) {
                d->restore_trial_state();
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
            ++stalled_iters;
        } else {
            stalled_iters = 0;
        }
        resto_stalled = stalled_iters >= 2;
    }

    const auto cleanup = [&](bool success) {
        if (success) {
            for_each_overlay_pair(resto_graph, outer_graph, [&](data &resto, data &outer) {
                sync_primal_state(resto, outer);
                sync_hard_duals(resto, outer);
            });
            clear_phase_graph_override();
            outer_graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_val, true);
                solver::ineq_soft::initialize(d);
                solver::restoration::reset_equality_duals(*d, settings.restoration.constr_mult_reset_threshold);
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
        } else {
            clear_phase_graph_override();
        }
        settings.in_restoration = false;
        if (!success) {
            outer_graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
        }
    };

    cleanup(resto_accept);
    kkt_info result = compute_kkt_info();
    result.num_iter = kkt_rest.num_iter;
    result.ls_steps = kkt_rest.ls_steps;
    if (resto_accept) {
        result.result = iter_result_t::success;
    } else if (resto_stalled && result.inf_dual_res < settings.dual_tol && result.inf_prim_res > settings.prim_tol) {
        result.result = iter_result_t::infeasible_stationary;
    } else {
        result.result = iter_result_t::restoration_failed;
    }
    if (settings.verbose) {
        fmt::print("=== leave restoration: {} ===\n\n",
                   resto_accept ? "success" :
                                  (result.result == iter_result_t::infeasible_stationary ? "infeasible_stationary" : "failed"));
    }
    return result;
}

} // namespace moto
