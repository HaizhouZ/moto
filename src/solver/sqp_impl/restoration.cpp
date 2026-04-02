#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_runtime.hpp>

namespace moto {

ns_sqp::kkt_info ns_sqp::restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls) {
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }

    auto &graph = solver_graph();
    const scalar_t mu_resto = std::max(settings.ipm.mu, kkt_before.inf_prim_res);
    settings.in_restoration = true;

    const auto prox_eps = scalar_t(1.0);
    for (data &node : graph.nodes()) {
        auto aux = std::make_unique<ns_riccati_data::restoration_aux_data>();
        aux->rho_eq = settings.restoration.rho_eq;
        aux->rho_ineq = settings.restoration.rho_ineq;
        aux->rho_u = settings.restoration.rho_u;
        aux->rho_y = settings.restoration.rho_y;
        aux->lambda_reg = settings.restoration.lambda_reg;
        aux->mu_bar = mu_resto;
        aux->verbose = settings.verbose;
        aux->outer_dual_backup = node.dense().dual_;
        aux->outer_dual_backup_valid = true;
        aux->u_ref = node.value(__u);
        aux->y_ref = node.value(__y);
        aux->sigma_u_sq.resizeLike(aux->u_ref);
        aux->sigma_y_sq.resizeLike(aux->y_ref);
        if (aux->u_ref.size() > 0) {
            aux->sigma_u_sq = aux->u_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }
        if (aux->y_ref.size() > 0) {
            aux->sigma_y_sq = aux->y_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }
        node.aux_ = std::move(aux);
    }

    graph.for_each_parallel([this](data *d) {
        assemble_restoration_problem(d, node_data::update_mode::eval_val);
        assemble_restoration_problem(d, node_data::update_mode::eval_raw_derivatives);
    });

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
                                             /*do_refinement=*/false,
                                             /*gauss_newton=*/false);
        kkt_rest.num_iter = kkt_before.num_iter + i_rest + 1;
        kkt_rest.ls_steps = rls.step_cnt;

        if (action == line_search_action::accept) {
            graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_val, true);
            });
            kkt_outer_trial = compute_kkt_info_for_phase(iteration_phase::normal, false);
            graph.for_each_parallel([this, &kkt_rest](data *d) {
                solver::restoration::update_mu_bar(*d, settings.ipm,
                                                   settings.ipm.mu_monotone_fraction_threshold,
                                                   settings.ipm.mu_monotone_factor,
                                                   kkt_rest.inf_prim_res,
                                                   kkt_rest.inf_dual_res);
            });
            graph.for_each_parallel([this](data *d) {
                assemble_restoration_problem(d, node_data::update_mode::eval_val);
                assemble_restoration_problem(d, node_data::update_mode::eval_raw_derivatives);
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
            ++stalled_iters;
        } else {
            stalled_iters = 0;
        }
        resto_stalled = stalled_iters >= 2;
    }

    const auto cleanup = [&](bool success) {
        graph.for_each_parallel([&](data *d) {
            if (dynamic_cast<ns_riccati_data::restoration_aux_data *>(d->aux_.get()) == nullptr) {
                return;
            }
            solver::restoration::cleanup_restoration_stage(*d,
                                                           success,
                                                           settings.restoration.bound_mult_reset_threshold,
                                                           settings.restoration.constr_mult_reset_threshold);
            d->aux_.reset();
        });

        settings.in_restoration = false;
        graph.for_each_parallel([this](data *d) {
            d->update_approximation(node_data::update_mode::eval_derivatives, true);
        });
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
    return result;
}

} // namespace moto
