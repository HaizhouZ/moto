/**
 * @file scaling.cpp
 * @brief Gradient- and equilibrium-based Jacobian scaling for the ns_sqp solver.
 *
 * Two strategies:
 *   gradient    – each constraint row is scaled so its Jacobian inf-norm ≈ 1, and
 *                 the cost gradient is scaled by the reciprocal of its inf-norm.
 *   equilibrium – iterative Ruiz / Sinkhorn balancing: alternately normalise rows
 *                 and columns until the per-row and per-column inf-norms are all ≈ 1.
 *
 * The scale vectors are cached in ns_sqp::data::scale_c_ / scale_p_ and reused
 * across iterations unless the KKT residuals grow beyond configurable thresholds.
 *
 * All scaling is applied *in-place* to lag_data (v_, jac_, cost jac_) so the
 * downstream Riccati factorization sees a well-conditioned QP.  After the solve,
 * unscale_duals() reverses the scaling on the dual variables so that the next
 * evaluation of the full KKT residuals is in original (physical) units.
 */
#include <moto/solver/ns_sqp.hpp>

namespace moto {

// ────────────────────────────────────────────────────────────────────────────
// compute_and_apply_scaling
// ────────────────────────────────────────────────────────────────────────────

void ns_sqp::reset_scaling() {
    auto &graph = active_data();
    for (data *d : graph.nodes()) {
        for (field_t cf : constr_fields)
            d->scale_c_[cf].resize(0);
        d->scale_p_.fill(1.);
        d->scaling_applied_ = false;
    }
}

void ns_sqp::compute_and_apply_scaling(const kkt_info &info) {
    const auto &sc = settings.scaling;
    auto &graph = active_data();
    if (sc.mode == scaling_settings::mode_t::none)
        return;

    // ── Decide whether to recompute scale vectors ────────────────────────────
    // Recompute only when the primal step from the last iteration was large
    // (inf_prim_step >= 1/update_ratio_threshold), meaning the Jacobians have
    // changed significantly.  Near convergence (small steps) the cached scales
    // remain valid and recomputing is wasteful.
    // Force recompute on the first call (scale_c_ still empty for some nodes).
    bool needs_recompute = (info.step.inf_prim_step >= 1. / sc.update_ratio_threshold);
    // Also recompute if no cached scales exist yet (first call after reset)
    if (!needs_recompute) {
        for (const data *d : graph.nodes()) {
            for (field_t cf : hard_constr_fields_non_dyn) {
                if (d->dense().approx_[cf].v_.size() > 0 && d->scale_c_[cf].size() == 0) {
                    needs_recompute = true;
                    break;
                }
            }
            if (needs_recompute)
                break;
        }
    }

    // Only __eq_x / __eq_xu are scaled; __dyn is excluded because approx_[__dyn].jac_[__y]
    // is aliased to f_y_ in dense_dynamics — scaling it in-place corrupts the LU used by
    // compute_project_jacobians (proj_f_x_, proj_f_u_) and apply_jac_y_inverse_transpose.
    // Inequality constraints (IPM) are also excluded: their Jacobians and duals are aliased
    // into ipm_constr and managed internally by the IPM.
    const auto apply_node_scaling = [](data *d) {
        for (field_t cf : hard_constr_fields_non_dyn) {
            const auto &s = d->scale_c_[cf];
            if (s.size() == 0)
                continue;
            auto &approx = d->dense().approx_[cf];
            approx.v_.array() *= s.array();
            d->scale_constraint_jacobian(cf, s);
        }
        d->scaling_applied_ = true;
    };
    const auto compute_node_scaling = [&](data *d) {
        const scalar_t min_s = sc.min_scale;

        for (field_t cf : hard_constr_fields_non_dyn) {
            auto &approx = d->dense().approx_[cf];
            int m = (int)approx.v_.size();
            if (m == 0) {
                d->scale_c_[cf].resize(0);
                continue;
            }

            vector &s = d->scale_c_[cf];
            s.resize(m);

            if (sc.mode == scaling_settings::mode_t::gradient) {
                s.setZero();
                d->constraint_row_infnorms(cf, s);
                // Widen scale for rows with large residuals too
                s = s.cwiseMax(approx.v_.cwiseAbs());
                for (int i = 0; i < m; ++i)
                    s[i] = 1. / std::max(min_s, s[i]);

            } else { // equilibrium
                s.setOnes();
                for (size_t iter = 0; iter < sc.equilibrium_iters; ++iter) {
                    // Work on a temporary copy so we don't touch approx in-place here.
                    vector row_norms = s.cwiseAbs().cwiseProduct(approx.v_.cwiseAbs());
                    d->constraint_row_infnorms(cf, row_norms, &s);
                    for (int i = 0; i < m; ++i)
                        s[i] /= std::max(min_s, row_norms[i]);
                }
                for (int i = 0; i < m; ++i)
                    s[i] = std::max(min_s, s[i]);
            }
        }

        // Cost gradient scaling is intentionally disabled.
        // Q_y is propagated across stages during the backward Riccati recursion.
        // Scaling Q_y in-place at each stage would therefore contaminate the
        // cross-stage first-order value terms across the horizon.
        d->scale_p_.fill(1.);
    };

    solver::for_each(solver::par, graph, [&](data *d) {
        if (needs_recompute) {
            compute_node_scaling(d);
        }
        apply_node_scaling(d);
    });
}

// ────────────────────────────────────────────────────────────────────────────
// unscale_duals
// Reverse in-place scaling so that:
//   (a) constraint residuals and Jacobians are back in original units, and
//   (b) dual variable *steps* are back in original units.
//
// Scaling applied s·J and s·v to constraints, so the solver sees (s·J)^T Δλ_scaled = -∇f.
// Original: J^T Δλ_orig = -∇f.  Therefore Δλ_orig = s · Δλ_scaled.
// Only trial_dual_step is unscaled; dense_->dual_[f] holds the accumulated λ from previous
// iterations and is already in original units — do NOT touch it here.
// ────────────────────────────────────────────────────────────────────────────

void ns_sqp::unscale_duals() {
    auto &graph = active_data();
    solver::for_each(solver::par, graph, [this](data *d) { unscale_duals(d); });
}

void ns_sqp::unscale_duals(data *d) {
    if (!d->scaling_applied_)
        return;
    // Only __eq_x / __eq_xu were scaled — reverse those only.
    // __dyn excluded: approx_[__dyn].jac_[__y] aliases f_y_ used in LU.
    // dense_->dual_[__ineq_*] aliases ipm_constr::multiplier_ (IPM-managed).
    // approx_[__ineq_*].jac_[] aliases ipm_constr::jac_[] (IPM barrier gradient).
    for (field_t cf : hard_constr_fields_non_dyn) {
        const auto &s = d->scale_c_[cf];
        if (s.size() == 0)
            continue;
        auto &approx = d->dense().approx_[cf];
        // Reverse constraint residual scaling: v_scaled = s * v_orig → v_orig = v_scaled / s
        if (approx.v_.size() > 0)
            approx.v_.array() /= s.array();
        // Reverse constraint Jacobian row scaling
        const vector inverse_scale = s.cwiseInverse();
        d->scale_constraint_jacobian(cf, inverse_scale);
    }
    // Unscale the dual *step* for __eq_x / __eq_xu only.
    // trial_dual_step[cf] is computed in the scaled system:
    //   (s·J)^T λ_scaled = -∇f  →  λ_orig_step = s · λ_scaled_step.
    // The current dual dual_[cf] was set in original units in a previous iteration
    // and must NOT be re-scaled here.
    for (field_t cf : hard_constr_fields_non_dyn) {
        const auto &s = d->scale_c_[cf];
        if (s.size() == 0)
            continue;
        auto &dlam = d->trial_dual_step[cf];
        if (dlam.size() > 0)
            dlam.array() *= s.array();
    }
    d->scaling_applied_ = false;
}

} // namespace moto
