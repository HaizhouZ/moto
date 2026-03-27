#include <moto/solver/ns_sqp.hpp>

namespace moto {

ns_sqp::kkt_info ns_sqp::restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls) {
    using namespace solver::ns_riccati;
    const auto &rs = settings.restoration;

    if (settings.verbose)
        fmt::print("=== Entering restoration phase (inf_prim_res: {:.3e}) ===\n",
                   kkt_before.inf_prim_res);

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
    std::vector<node_prox> prox_data;
    prox_data.reserve(graph_.nodes().size());
    for (data &n : graph_.nodes()) {
        auto &p = prox_data.emplace_back();
        p.u_ref = n.value(__u);
        p.y_ref = n.value(__y);
        p.sigma_u_sq = p.u_ref.array().abs().max(prox_eps).inverse().square().min(1.);
        p.sigma_y_sq = p.y_ref.array().abs().max(prox_eps).inverse().square().min(1.);

        // Mark restoration mode; rho_eq controls GN dual regularization strength.
        auto *aux = new ns_riccati_data::restoration_aux_data();
        aux->rho_eq = rs.rho_eq;
        n.aux_.reset(aux);
    }

    // Dedicated restoration filter: tracks J_rest = ½‖F_0‖² + ½‖s_c_0_k‖² instead of
    // the original running cost, so filter progress reflects feasibility improvement.
    // constr_vio_min=0 means pure filter mode (no Armijo switching).
    filter_linesearch_data rls;
    rls.constr_vio_min = 0.; //// todo: maybe allow nonzero constr_vio_min to enforce some minimum restoration progress before accepting a step?

    kkt_info kkt_rest = kkt_before;

    for (size_t i_rest = 0; i_rest < rs.max_iter; ++i_rest) {
        // Write proximal gradient and Hessian directly into jac_modification_ and
        // hessian_modification_ after update_approximation has zeroed them.
        // merge_jacobian_modification (inside ns_factorization) will fold them into Q_u/Q_y.
        size_t node_idx = 0;
        for (data &n : graph_.nodes()) {
            const auto &p = prox_data[node_idx++];
            n.dense().jac_modification_[__u].noalias() +=
                rho_u * (p.sigma_u_sq.array() * (n.value(__u) - p.u_ref).array()).matrix().transpose();
            n.dense().jac_modification_[__y].noalias() +=
                rho_y * (p.sigma_y_sq.array() * (n.value(__y) - p.y_ref).array()).matrix().transpose();
            n.dense().primal_prox_hess_diagonal_[__u].diagonal() += rho_u * p.sigma_u_sq;
            n.dense().primal_prox_hess_diagonal_[__y].diagonal() += rho_y * p.sigma_y_sq;
        }

        line_search_action rest_action = sqp_iter(rls, kkt_rest,
                                                  /*do_scaling=*/false, /*do_refinement=*/true,
                                                  /*gauss_newton=*/true);

        kkt_rest.num_iter = i_rest + 1;
        kkt_rest.ls_steps = rls.step_cnt;

        if (settings.verbose) {
            print_stats(kkt_rest);
        }

        // Exit when line search accepted a step (genuine progress in the restoration
        // sub-problem) OR feasibility has improved sufficiently relative to the entry point.
        // If line search completely failed (stop) and neither criterion is met, give up.
        bool prim_improved = (kkt_rest.inf_prim_res < rs.restoration_improvement_frac * kkt_before.inf_prim_res);
        if (rest_action != line_search_action::stop || prim_improved) {
            if (settings.verbose)
                fmt::print("=== Restoration succeeded (inf_prim_res: {:.3e}) ===\n",
                           kkt_rest.inf_prim_res);
            break;
        }
        // line search failed and no sufficient feasibility progress — give up
        break;
    }

    graph_.for_each_parallel([](data *d) {
        d->aux_.reset();
    });
    ls.update_filter(kkt_rest, settings);

    if (settings.verbose)
        fmt::print("=== Leaving restoration phase (inf_prim_res: {:.3e}) ===\n",
                   kkt_rest.inf_prim_res);

    return kkt_rest;
}

} // namespace moto
