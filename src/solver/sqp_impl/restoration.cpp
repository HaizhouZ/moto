#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_runtime.hpp>

namespace moto {

ns_sqp::kkt_info ns_sqp::restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls) {
    using namespace solver::ns_riccati;
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        throw std::runtime_error("restoration mode is incompatible with merit_backtracking");
    }
    const auto &rs = settings.restoration;
    auto &graph = solver_graph();
    bool resto_accept = false;
    bool resto_converge = false;
    settings.in_restoration = true;
    const scalar_t mu_regular = settings.ipm.mu;
    const scalar_t mu_resto = std::max(mu_regular, kkt_before.inf_prim_res);
    settings.ipm.mu = mu_resto;
    fmt::print("[resto]: triggered restoration with rho_u: {:.3e}, rho_y: {:.3e}, mu_bar: {:.3e}, lambda_reg: {:.3e}\n",
               rs.rho_u, rs.rho_y, mu_resto, rs.lambda_reg);

    // Snapshot proximal reference points and per-component primal scaling.
    // sigma[i] = 1/max(|ref[i]|, 1) so the proximal cost is on a percentage level.
    const scalar_t prox_eps = 1.0;
    auto &all_nodes = graph.flatten_nodes();
    const auto cleanup_restoration_state = [&] {
        for (data *d : all_nodes) {
            if (auto *aux = dynamic_cast<ns_riccati_data::restoration_aux_data *>(d->aux_.get()); aux != nullptr && aux->initialized) {
                d->last_restoration_.valid = true;
                d->last_restoration_.mu_bar = aux->mu_bar;
                d->last_restoration_.p = aux->elastic.p;
                d->last_restoration_.n = aux->elastic.n;
                d->last_restoration_.nu_p = aux->elastic.nu_p;
                d->last_restoration_.nu_n = aux->elastic.nu_n;
                d->last_restoration_.lambda = solver::restoration::gather_lambda(*d);
            }
            d->aux_.reset();
        }
        settings.ipm.mu = mu_regular;
        settings.in_restoration = false;
    };
    for (data *np : all_nodes) {
        data &n = *np;
        // Mark restoration mode; presolve/rollout will initialize and update
        // the explicit local elastic block using mu_bar, rho_eq, and lambda_reg.
        auto *aux = new ns_riccati_data::restoration_aux_data();
        aux->rho_eq = rs.rho_eq;
        aux->lambda_reg = rs.lambda_reg;
        aux->mu_bar = mu_resto;
        aux->verbose = settings.verbose;
        n.restoration_prox_.u_ref = n.value(__u);
        n.restoration_prox_.y_ref = n.value(__y);
        n.restoration_prox_.sigma_u_sq.resizeLike(n.restoration_prox_.u_ref);
        n.restoration_prox_.sigma_y_sq.resizeLike(n.restoration_prox_.y_ref);
        if (n.restoration_prox_.u_ref.size() > 0) {
            n.restoration_prox_.sigma_u_sq =
                n.restoration_prox_.u_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }
        if (n.restoration_prox_.y_ref.size() > 0) {
            n.restoration_prox_.sigma_y_sq =
                n.restoration_prox_.y_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }
        n.aux_.reset(aux);

        // Reset original equality multipliers when entering restoration. The
        // explicit elastic lambda_c will be initialized from the IPOPT-style
        // local subproblem on the first restoration factorization.
        n.dense().dual_[__dyn].setZero();
        n.dense().dual_[__eq_x].setZero();
        n.dense().dual_[__eq_xu].setZero();
    }

    graph.for_each_parallel([this](data *d) {
        d->update_approximation(node_data::update_mode::eval_val,
                                /*include_original_cost=*/false);
        solver::restoration::prepare_current_constraint_stack(*d);
        solver::restoration::initialize_stage(*d);
        d->update_approximation(node_data::update_mode::eval_raw_derivatives,
                                /*include_original_cost=*/false);
        solver::restoration::assemble_resto_base_lagrangian(*d);
        solver::restoration::add_resto_prox_term(
            *d, settings.restoration.rho_u, settings.restoration.rho_y);
    });

    // Reuse the outer filter history, but restoration itself does not add new
    // entries to that filter.
    ls.augment_filter_for_restoration_start(kkt_before, settings);
    filter_linesearch_data rls;
    rls.constr_vio_min = ls.constr_vio_min;
    rls.resto.reset();

    kkt_info kkt_rest = compute_kkt_info();
    kkt_info resto_kkt = compute_restoration_kkt_info();
    scalar_t phi_prev = resto_kkt.inf_prim_res;
    size_t stalled_iters = 0;
    constexpr size_t min_iters_before_infeasible = 2;
    constexpr scalar_t stall_tol = 1e-12;
    const size_t max_resto_iters =
        std::min(rs.max_iter,
                 settings.max_iter > kkt_before.num_iter
                     ? settings.max_iter - kkt_before.num_iter
                     : size_t(0));
    for (size_t i_rest = 0; i_rest < max_resto_iters; ++i_rest) {
        line_search_action rest_action = sqp_iter(rls, kkt_rest,
                                                  /*do_scaling=*/false,
                                                  /*do_refinement=*/true,
                                                  /*gauss_newton=*/true);
        const auto resto_eval = compute_restoration_info();
        resto_kkt = compute_restoration_kkt_info();
        rls.resto.current = resto_kkt;

        kkt_rest.num_iter = kkt_before.num_iter + i_rest + 1;
        kkt_rest.ls_steps = rls.step_cnt;

        if (settings.verbose) {
            print_stats(kkt_rest);
            fmt::print("[resto metrics] theta_R: {:.3e}, phi_R: {:.3e}, dual_R: {:.3e} (w={:.3e}, rp={:.3e}, rn={:.3e}), comp_R: {:.3e} (p={:.3e}, n={:.3e})\n",
                       resto_eval.theta, resto_eval.phi, resto_eval.dual,
                       resto_eval.dual_w, resto_eval.dual_local_p, resto_eval.dual_local_n,
                       resto_eval.comp, resto_eval.comp_p, resto_eval.comp_n);
            print_dual_res_breakdown();
        }

        if (rest_action == line_search_action::failure) {
            if (settings.verbose) {
                fmt::print("[resto]: line search failed at restoration iter {} "
                           "(theta_R={:.3e}, dual_R={:.3e}, comp_R={:.3e}), continue restoration\n",
                           i_rest + 1,
                           resto_eval.theta,
                           resto_eval.dual,
                           resto_eval.comp);
            }
            ++stalled_iters;
        }

        bool prim_improved =
            (rls.resto.current.inf_prim_res < rs.restoration_improvement_frac * kkt_before.inf_prim_res);
        if (rest_action == line_search_action::accept) {
            const bool outer_accept = outer_filter_accepts(ls, kkt_rest, kkt_before);
            if (settings.verbose) {
                fmt::print("[resto]: outer acceptability after accepted restoration step: {} (prim_improved={})\n",
                           outer_accept ? "accepted" : "rejected",
                           prim_improved ? "yes" : "no");
            }
            if (outer_accept && prim_improved) {
                resto_accept = true;
                break;
            }
        }

        const scalar_t prim_delta = std::abs(phi_prev - rls.resto.current.inf_prim_res);
        const bool stalled = prim_delta <= stall_tol * std::max<scalar_t>(1., phi_prev);
        if (stalled) {
            ++stalled_iters;
        } else {
            stalled_iters = 0;
        }

        const scalar_t dual_local =
            std::max(resto_eval.dual_local_p, resto_eval.dual_local_n);
        if (dual_local < settings.dual_tol &&
            !(resto_eval.theta < settings.prim_tol &&
              resto_eval.comp < settings.comp_tol) &&
            stalled_iters >= min_iters_before_infeasible) {
            resto_converge = true;
            break;
        }

        phi_prev = rls.resto.current.inf_prim_res;
    }

    cleanup_restoration_state();

    if (!resto_accept) {
        if (resto_converge) {
            fmt::println("[resto]: dual residual below tolerance, converging to infeasible stationary point, exiting restoration");
            kkt_rest.result = iter_result_t::infeasible_stationary;
        } else {
            fmt::println("[resto]: restoration failed to make sufficient progress, exiting restoration");
            kkt_rest.result = iter_result_t::restoration_failed;
        }
    } else {
        fmt::println("[resto]: restoration successful, exiting restoration");
    }

    return kkt_rest;
}

} // namespace moto
