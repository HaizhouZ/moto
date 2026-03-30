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
#include <moto/spmm/sparse_mat.hpp>

namespace moto {

// ────────────────────────────────────────────────────────────────────────────
// Helpers for efficient in-place row-/column-scaling of sparse_mat panels
// ────────────────────────────────────────────────────────────────────────────

namespace {

/// Row-scale a sparse_mat in-place: row i ← s[i] * row i.
/// Avoids dense materialisation by iterating over stored panels.
void row_scale_inplace(sparse_mat &mat, const vector &s) {
    // dense panels
    for (auto &panel : mat.dense_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)s.size())
                panel.data_.row(local_r) *= s[glob_r];
        }
    }
    // diagonal panels: data_ is a column vector (diag entries)
    for (auto &panel : mat.diag_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)s.size())
                panel.data_[local_r] *= s[glob_r];
        }
    }
    // eye panels: data_ is a vector of ones; scale → diagonal scale
    for (auto &panel : mat.eye_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)s.size())
                panel.data_[local_r] *= s[glob_r];
        }
    }
}

/// Update the per-row inf-norms of a sparse_mat *into* norms[row_st..row_ed),
/// using norms(row) = max(norms(row), |panel entry|).
void accumulate_row_infnorms(const sparse_mat &mat, Eigen::Ref<vector> norms) {
    for (const auto &panel : mat.dense_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)norms.size()) {
                scalar_t v = panel.data_.row(local_r).cwiseAbs().maxCoeff();
                if (v > norms[glob_r])
                    norms[glob_r] = v;
            }
        }
    }
    for (const auto &panel : mat.diag_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)norms.size()) {
                scalar_t v = std::abs(panel.data_[local_r]);
                if (v > norms[glob_r])
                    norms[glob_r] = v;
            }
        }
    }
    for (const auto &panel : mat.eye_panels_) {
        for (int local_r = 0; local_r < panel.rows_; ++local_r) {
            int glob_r = panel.row_st_ + local_r;
            if (glob_r < (int)norms.size()) {
                // eye panel: |entry| = 1
                if (1. > norms[glob_r])
                    norms[glob_r] = 1.;
            }
        }
    }
}

} // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
// compute_and_apply_scaling
// ────────────────────────────────────────────────────────────────────────────

void ns_sqp::reset_scaling() {
    for (data &d : graph_.nodes()) {
        for (field_t cf : constr_fields)
            d.scale_c_[cf].resize(0);
        d.scale_p_.fill(1.);
        d.scaling_applied_ = false;
    }
}

void ns_sqp::compute_and_apply_scaling(const kkt_info &kkt) {
    const auto &sc = settings.scaling;
    if (sc.mode == scaling_settings::mode_t::none)
        return;

    // ── Decide whether to recompute scale vectors ────────────────────────────
    // Recompute only when the primal step from the last iteration was large
    // (inf_prim_step >= 1/update_ratio_threshold), meaning the Jacobians have
    // changed significantly.  Near convergence (small steps) the cached scales
    // remain valid and recomputing is wasteful.
    // Force recompute on the first call (scale_c_ still empty for some nodes).
    bool needs_recompute = (kkt.inf_prim_step >= 1. / sc.update_ratio_threshold);
    // Also recompute if no cached scales exist yet (first call after reset)
    if (!needs_recompute) {
        for (const data &d : graph_.nodes()) {
            for (field_t cf : hard_constr_fields_non_dyn) {
                if (d.dense().approx_[cf].v_.size() > 0 && d.scale_c_[cf].size() == 0) {
                    needs_recompute = true;
                    break;
                }
            }
            if (needs_recompute)
                break;
        }
    }

    // ── Recompute scale vectors if needed ────────────────────────────────────
    if (needs_recompute) {
        graph_.for_each_parallel([&](data *d) {
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
                    for (field_t pf : primal_fields)
                        if (!approx.jac_[pf].is_empty())
                            accumulate_row_infnorms(approx.jac_[pf], s);
                    // Widen scale for rows with large residuals too
                    s = s.cwiseMax(approx.v_.cwiseAbs());
                    for (int i = 0; i < m; ++i)
                        s[i] = 1. / std::max(min_s, s[i]);

                } else { // equilibrium
                    s.setOnes();
                    for (size_t iter = 0; iter < sc.equilibrium_iters; ++iter) {
                        // Accumulate scaled row inf-norms via row_scale_inplace helper:
                        // apply s, accumulate norms, then divide s by norms.
                        // Work on a temporary copy so we don't touch approx in-place here.
                        vector row_norms = s.cwiseAbs().cwiseProduct(approx.v_.cwiseAbs());
                        for (field_t pf : primal_fields) {
                            if (approx.jac_[pf].is_empty())
                                continue;
                            // Compute s[i] * ||J_row_i||_inf for each row using stored panels.
                            for (const auto &panel : approx.jac_[pf].dense_panels_) {
                                for (int local_r = 0; local_r < panel.rows_; ++local_r) {
                                    int i = panel.row_st_ + local_r;
                                    if (i >= m) continue;
                                    scalar_t rn = s[i] * panel.data_.row(local_r).cwiseAbs().maxCoeff();
                                    if (rn > row_norms[i]) row_norms[i] = rn;
                                }
                            }
                            for (const auto &panel : approx.jac_[pf].diag_panels_) {
                                for (int local_r = 0; local_r < panel.rows_; ++local_r) {
                                    int i = panel.row_st_ + local_r;
                                    if (i >= m) continue;
                                    scalar_t rn = s[i] * std::abs(panel.data_[local_r]);
                                    if (rn > row_norms[i]) row_norms[i] = rn;
                                }
                            }
                            for (const auto &panel : approx.jac_[pf].eye_panels_) {
                                for (int local_r = 0; local_r < panel.rows_; ++local_r) {
                                    int i = panel.row_st_ + local_r;
                                    if (i >= m) continue;
                                    if (s[i] > row_norms[i]) row_norms[i] = s[i]; // |1| * s[i]
                                }
                            }
                        }
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
        });
    }

    // ── Always apply the (possibly freshly computed) scale vectors ───────────
    // Only __eq_x / __eq_xu are scaled; __dyn is excluded because approx_[__dyn].jac_[__y]
    // is aliased to f_y_ in dense_dynamics — scaling it in-place corrupts the LU used by
    // compute_project_jacobians (proj_f_x_, proj_f_u_) and apply_jac_y_inverse_transpose.
    // Inequality constraints (IPM) are also excluded: their Jacobians and duals are aliased
    // into ipm_constr and managed internally by the IPM.
    graph_.for_each_parallel([&](data *d) {
        for (field_t cf : hard_constr_fields_non_dyn) {
            const auto &s = d->scale_c_[cf];
            if (s.size() == 0)
                continue;
            auto &approx = d->dense().approx_[cf];
            approx.v_.array() *= s.array();
            for (field_t pf : primal_fields)
                if (!approx.jac_[pf].is_empty())
                    row_scale_inplace(approx.jac_[pf], s);
        }
        d->scaling_applied_ = true;
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
    graph_.for_each_parallel([](data *d) {
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
            for (field_t pf : primal_fields)
                if (!approx.jac_[pf].is_empty())
                    row_scale_inplace(approx.jac_[pf], s.cwiseInverse());
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
    });
}

} // namespace moto
