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
    const scalar_t rho_u = settings.restoration.rho_u;
    const scalar_t rho_y = settings.restoration.rho_y;
    const scalar_t prox_eps = 1.0;

    // Per-node snapshots (sequential — nodes() is ordered and small relative to stage count)
    struct node_prox {
        vector u_ref, y_ref;
        vector sigma_u_sq, sigma_y_sq; // rho-scaled: (1/max(|ref|,1))^2
    };
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
    std::vector<node_prox> prox_data;
    prox_data.reserve(all_nodes.size());
    for (data *np : all_nodes) {
        data &n = *np;
        auto &p = prox_data.emplace_back();
        p.u_ref = n.value(__u);
        p.y_ref = n.value(__y);
        p.sigma_u_sq.resizeLike(p.u_ref);
        p.sigma_y_sq.resizeLike(p.y_ref);
        if (p.u_ref.size() > 0) {
            p.sigma_u_sq = p.u_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }
        if (p.y_ref.size() > 0) {
            p.sigma_y_sq = p.y_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        }

        // Mark restoration mode; presolve/rollout will initialize and update
        // the explicit local elastic block using mu_bar, rho_eq, and lambda_reg.
        auto *aux = new ns_riccati_data::restoration_aux_data();
        aux->rho_eq = rs.rho_eq;
        aux->lambda_reg = rs.lambda_reg;
        aux->mu_bar = mu_resto;
        aux->verbose = settings.verbose;
        n.aux_.reset(aux);

        // Reset original equality multipliers when entering restoration. The
        // explicit elastic lambda_c will be initialized from the IPOPT-style
        // local subproblem on the first restoration factorization.
        n.dense().dual_[__dyn].setZero();
        n.dense().dual_[__eq_x].setZero();
        n.dense().dual_[__eq_xu].setZero();
    }

    // Reuse the outer filter history, but restoration itself does not add new
    // entries to that filter.
    filter_linesearch_data rls = ls;

    kkt_info kkt_rest = compute_kkt_info();
    scalar_t phi_prev = kkt_rest.inf_prim_res;
    size_t stalled_iters = 0;
    constexpr size_t min_iters_before_infeasible = 2;
    constexpr scalar_t stall_tol = 1e-12;

    for (size_t i_rest = 0; kkt_before.num_iter + i_rest + 1 <= settings.max_iter; ++i_rest) {
        // Write proximal gradient corrections into lag_jac_corr_.
        // activate_lag_jac_corr() inside ns_factorization() will fold them
        // into the active stage gradient seen by the linear solve.
        size_t node_idx = 0;
        for (data *np : all_nodes) {
            data &n = *np;
            const auto &p = prox_data[node_idx++];
            if (p.u_ref.size() > 0) {
                n.dense().lag_jac_corr_[__u].noalias() +=
                    rho_u * (p.sigma_u_sq.array() * (n.value(__u) - p.u_ref).array()).matrix().transpose();
            }
            if (p.y_ref.size() > 0) {
                n.dense().lag_jac_corr_[__y].noalias() +=
                    rho_y * (p.sigma_y_sq.array() * (n.value(__y) - p.y_ref).array()).matrix().transpose();
            }
        }

        line_search_action rest_action = sqp_iter(rls, kkt_rest,
                                                  /*do_scaling=*/false,
                                                  /*do_refinement=*/true,
                                                  /*gauss_newton=*/true);

        kkt_rest.num_iter = kkt_before.num_iter + i_rest + 1;
        kkt_rest.ls_steps = rls.step_cnt;

        if (settings.verbose) {
            print_stats(kkt_rest);
        }

        if (rest_action == line_search_action::failure) {
            if (settings.verbose) {
                fmt::print("[resto]: line search failed at restoration iter {} "
                           "(prim_res={:.3e}, dual_res={:.3e}, comp_res={:.3e}), continue restoration\n",
                           i_rest + 1,
                           kkt_rest.inf_prim_res,
                           kkt_rest.inf_dual_res,
                           kkt_rest.inf_comp_res);
            }
            ++stalled_iters;
        }

        bool prim_improved = (kkt_rest.inf_prim_res < rs.restoration_improvement_frac * kkt_before.inf_prim_res);
        if (rest_action == line_search_action::accept && prim_improved) {
            resto_accept = true;
            break;
        }

        const scalar_t prim_delta = std::abs(phi_prev - kkt_rest.inf_prim_res);
        const bool stalled = prim_delta <= stall_tol * std::max<scalar_t>(1., phi_prev);
        if (stalled) {
            ++stalled_iters;
        } else {
            stalled_iters = 0;
        }

        if (kkt_rest.inf_dual_res < settings.dual_tol &&
            !(kkt_rest.inf_prim_res < settings.prim_tol &&
              kkt_rest.inf_comp_res < settings.comp_tol) &&
            stalled_iters >= min_iters_before_infeasible) {
            resto_converge = true;
            break;
        }

        phi_prev = kkt_rest.inf_prim_res;
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
