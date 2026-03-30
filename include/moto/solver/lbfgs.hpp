#ifndef MOTO_SOLVER_LBFGS_HPP
#define MOTO_SOLVER_LBFGS_HPP

#include <moto/core/fwd.hpp>
#include <moto/spmm/sparse_mat.hpp>
#include <Eigen/Cholesky>
#include <deque>

namespace moto {
namespace solver {

/// Settings for the structured L-BFGS Hessian correction.
/// Defined here so ns_riccati_data can hold a pointer without including ns_sqp.hpp.
struct lbfgs_settings {
    bool enabled = false;          ///< whether to use structured L-BFGS Hessian correction
    size_t max_pairs = 10;         ///< number of (s, y#) curvature pairs to keep per stage
    scalar_t min_curvature = 1e-8; ///< skip pairs where y#·s < min_curvature * ‖s‖²
};

/// Per-stage L-BFGS history for the **structured** approximation.
///
/// Pairs (s, y#) are stored in the original u-space (before null-space projection).
///
/// y# = (merit_jac_new[u] - merit_jac_old[u]) - (Q_uu + Q_uu_mod) * s
///
/// This removes the known Hessian contributions (approx_order=second cost/constraint Hessians
/// and the IPM/PMM diagonal Hessian terms) so that A_k approximates only the unknown part
/// of the Lagrangian Hessian:
///   B_k = C_k + H_IPM_k + A_k
///
/// At each factorization A_k is applied additively to Q_zz after the Riccati backward pass
/// has accumulated Z_y^T·V_yy·Z_y (constrained) or F_u^T·V_yy·F_u (unconstrained).
struct lbfgs_history {
    struct pair {
        vector s; ///< accepted primal step in u-space: alpha * trial_prim_step[__u]
        vector y; ///< structured gradient change y# in u-space (known Hessian contribution removed)
    };
    std::deque<pair> pairs;
    size_t max_pairs = 10;
    scalar_t min_curvature = scalar_t(1e-8);

    void push(vector s_in, vector y_in) {
        if (pairs.size() == max_pairs)
            pairs.pop_front();
        pairs.push_back({std::move(s_in), std::move(y_in)});
    }

    void clear() { pairs.clear(); }
    bool empty() const { return pairs.empty(); }
};

// ─── Implementation detail ────────────────────────────────────────────────────

namespace detail {

/// Apply a single damped BFGS rank-2 update to B in-place.
///
/// Satisfies the secant condition B_new * s = y_eff (after Powell damping).
/// Skips the update and returns false when curvature is too weak, the current
/// Hessian estimate has zero curvature along s, or the candidate is not PD.
///
/// Powell damping: when the raw curvature y·s < damping_fraction * s·B·s, we
/// replace y with y_eff = θ·y + (1-θ)·B·s, choosing θ so that
/// y_eff·s = damping_fraction * s·B·s > 0.  This keeps B PD without discarding
/// the step entirely.
///
/// @param B              nz×nz Hessian matrix (modified in-place on success)
/// @param s              step vector (nz)
/// @param y              gradient-change vector (nz)
/// @param min_curvature  skip when y·s < min_curvature * ‖s‖²
/// @return true iff the update was applied
inline bool bfgs_rank2_update(matrix &B,
                               const vector &s,
                               const vector &y,
                               scalar_t min_curvature) {
    constexpr scalar_t damping_fraction = scalar_t(0.2);

    // Guard against corrupted pairs (NaN/Inf in step or gradient change).
    if (!s.allFinite() || !y.allFinite()) return false;

    const scalar_t sy = y.dot(s);
    const scalar_t ss = s.squaredNorm();
    if (ss < scalar_t(1e-20)) return false;
    if (sy < min_curvature * ss) return false;

    const vector Bs  = B * s;
    const scalar_t sBs = s.dot(Bs);
    if (sBs <= scalar_t(1e-20)) return false;

    // Powell damping: enforce y_eff·s >= damping_fraction * s·B·s
    vector    y_eff  = y;
    scalar_t  ys_eff = sy;
    if (sy < damping_fraction * sBs) {
        // θ chosen so that θ·sy + (1-θ)·sBs = damping_fraction·sBs
        const scalar_t theta = (scalar_t(1) - damping_fraction) * sBs / (sBs - sy);
        y_eff  = theta * y + (scalar_t(1) - theta) * Bs;
        ys_eff = y_eff.dot(s); // = damping_fraction * sBs  (by construction)
        if (ys_eff <= scalar_t(1e-20)) return false;
    }

    // Standard BFGS Hessian update:
    //   B_new = B - (B·s·s^T·B)/(s^T·B·s) + (y_eff·y_eff^T)/(y_eff^T·s)
    matrix B_candidate = B;
    B_candidate.noalias() -= (Bs * Bs.transpose()) / sBs;
    B_candidate.noalias() += (y_eff * y_eff.transpose()) / ys_eff;
    // Symmetrize to suppress floating-point asymmetry before the PD check
    B_candidate = scalar_t(0.5) * (B_candidate + B_candidate.transpose());

    // Reject if the candidate contains non-finite values (can arise from
    // very large Bs when B is ill-conditioned before the first update).
    if (!B_candidate.allFinite()) return false;

    // Only accept the update if the result is positive definite
    Eigen::LLT<matrix> llt(B_candidate);
    if (llt.info() != Eigen::Success) return false;

    B = std::move(B_candidate);
    return true;
}

} // namespace detail

// ─── Structured L-BFGS pair collection ───────────────────────────────────────

/// Collect a structured curvature pair (s_u, y#_u) and append to history.
///
/// The structured gradient change removes the known Hessian contributions so
/// that L-BFGS approximates only the unknown part A_k:
///
///   y# = (merit_jac_new[u] - merit_jac_old[u]) - (Q_uu + Q_uu_mod) * s_u
///
/// This leaves A_k to capture ∇²l_unknown + Σ λ_i·∇²h_i + Σ ν_j·∇²g_j,
/// exactly what the Riccati/null-space solver cannot represent analytically.
///
/// Q_uu and Q_uu_mod should be the known Hessian at the **new** point
/// (evaluated after update_approximation at the accepted step).
///
/// Powell damping is deferred to apply time (inside bfgs_rank2_update), where
/// the current A_k estimate is available as the running B matrix.
///
/// @param history          per-stage L-BFGS history (modified)
/// @param merit_jac_new_u  merit_jac_[__u] at the accepted new point (row_vector)
/// @param merit_jac_old_u  merit_jac_[__u] saved at the start of the SQP iteration
/// @param Q_uu             known cost/constraint second-order Hessian (u×u sparse)
/// @param Q_uu_mod         IPM/PMM Hessian modification (u×u sparse)
/// @param s_u              actual primal step applied: alpha * trial_prim_step[__u]
/// @param min_curvature    discard the pair when ‖s_u‖² is below this floor
inline void lbfgs_collect_structured_pair(lbfgs_history &history,
                                           const row_vector &merit_jac_new_u,
                                           const row_vector &merit_jac_old_u,
                                           const sparse_mat &Q_uu,
                                           const sparse_mat &Q_uu_mod,
                                           vector s_u,
                                           scalar_t min_curvature) {
    if (s_u.size() == 0) return;
    const scalar_t ss = s_u.squaredNorm();
    if (ss < scalar_t(1e-20)) return;

    // Full Lagrangian gradient change (column vector, u-space)
    vector y_full = (merit_jac_new_u - merit_jac_old_u).transpose();

    // Remove known Hessian contribution: y# = y_full - (Q_uu + Q_uu_mod) * s_u
    // sparse_mat::times<true> computes dst += M * rhs
    vector Cks = vector::Zero(s_u.size());
    Q_uu.times<true>(s_u, Cks);
    Q_uu_mod.times<true>(s_u, Cks);
    y_full.noalias() -= Cks;

    // Guard: reject obviously degenerate pairs before storing
    if (!y_full.allFinite() || !s_u.allFinite()) return;
    if (y_full.dot(s_u) < min_curvature * ss) return;

    history.push(std::move(s_u), std::move(y_full));
}

// ─── Structured L-BFGS apply functions ───────────────────────────────────────
//
// These functions build the unknown-Hessian estimate A_k from a scaled-identity
// initialisation (γ·I, where γ comes from the last valid pair's curvature) and
// apply sequential damped BFGS rank-2 updates using the stored (s, y#) pairs.
// The resulting A_k is then **added** to Q_zz:
//
//   Q_zz += A_k
//
// so the final matrix satisfies
//
//   Q_zz ≈ C_k + H_IPM_k + V_yy-contribution + A_k
//
// Call these AFTER the Riccati backward pass has finished accumulating V_yy
// into Q_zz and BEFORE llt_ns_.compute(Q_zz).

/// Constrained case: pairs are projected into the current null-space via Z_u.
///
/// @param Q_zz          nz×nz projected Hessian (modified in-place)
/// @param Z_u           nu×nz null-space basis for the current stage
/// @param history       L-BFGS history (u-space pairs)
inline void lbfgs_apply_structured_to_qzz(matrix &Q_zz,
                                            const matrix &Z_u,
                                            const lbfgs_history &history) {
    if (history.empty()) return;
    if (!Q_zz.allFinite()) return;

    const int nu = static_cast<int>(Z_u.rows());
    const int nz = static_cast<int>(Z_u.cols());
    const scalar_t min_curvature = history.min_curvature;

    // Estimate initial scaling γ from the last valid pair in null-space coords.
    scalar_t gamma = scalar_t(1);
    for (auto it = history.pairs.rbegin(); it != history.pairs.rend(); ++it) {
        if (it->s.size() != nu || it->y.size() != nu) continue;
        const vector s_p = Z_u.transpose() * it->s;
        const vector y_p = Z_u.transpose() * it->y;
        const scalar_t sy = y_p.dot(s_p);
        const scalar_t ss = s_p.squaredNorm();
        if (ss > scalar_t(1e-20) && sy > min_curvature * ss) {
            gamma = sy / ss;
            break;
        }
    }

    // Build A_k starting from γ·I_{nz}
    matrix B_A = matrix::Identity(nz, nz) * gamma;
    for (const auto &p : history.pairs) {
        if (p.s.size() != nu || p.y.size() != nu) continue;
        const vector s_p = Z_u.transpose() * p.s;
        const vector y_p = Z_u.transpose() * p.y;
        detail::bfgs_rank2_update(B_A, s_p, y_p, min_curvature);
    }

    // Symmetrize before adding to guard against accumulated asymmetry
    B_A = scalar_t(0.5) * (B_A + B_A.transpose());

    if (B_A.allFinite())
        Q_zz.noalias() += B_A;
}

/// Unconstrained case: Z_u = I_{nu}, so no null-space projection is needed.
///
/// @param Q_zz          nu×nu Hessian in u-space (modified in-place)
/// @param history       L-BFGS history (u-space pairs)
inline void lbfgs_apply_structured_to_qzz(matrix &Q_zz,
                                            const lbfgs_history &history) {
    if (history.empty()) return;
    if (!Q_zz.allFinite()) return;

    const int nu = static_cast<int>(Q_zz.rows());
    const scalar_t min_curvature = history.min_curvature;

    // Estimate initial scaling γ from the last valid pair.
    scalar_t gamma = scalar_t(1);
    for (auto it = history.pairs.rbegin(); it != history.pairs.rend(); ++it) {
        if (it->s.size() != nu || it->y.size() != nu) continue;
        const scalar_t sy = it->y.dot(it->s);
        const scalar_t ss = it->s.squaredNorm();
        if (ss > scalar_t(1e-20) && sy > min_curvature * ss) {
            gamma = sy / ss;
            break;
        }
    }

    // Build A_k starting from γ·I_{nu}
    matrix B_A = matrix::Identity(nu, nu) * gamma;
    for (const auto &p : history.pairs) {
        if (p.s.size() != nu || p.y.size() != nu) continue;
        detail::bfgs_rank2_update(B_A, p.s, p.y, min_curvature);
    }

    B_A = scalar_t(0.5) * (B_A + B_A.transpose());

    if (B_A.allFinite())
        Q_zz.noalias() += B_A;
}

// ─── Legacy full-gradient L-BFGS (kept for reference / non-structured use) ───
//
// These apply BFGS updates *starting from Q_zz itself* using the raw
// (unstructured) gradient change y = merit_jac_new - merit_jac_old.
// They are NOT used by the structured path; retained to avoid breaking any
// downstream code that may call them directly.

/// Constrained legacy apply (Z_u projection, starts from Q_zz as initial B).
inline void lbfgs_apply_to_qzz(matrix &Q_zz,
                                const matrix &Z_u,
                                const lbfgs_history &history,
                                scalar_t min_curvature = scalar_t(1e-8)) {
    if (history.empty()) return;
    if (!Q_zz.allFinite()) return;

    const int nu = static_cast<int>(Z_u.rows());
    matrix B = Q_zz;

    for (const auto &p : history.pairs) {
        if (p.s.size() != nu || p.y.size() != nu) continue;
        const vector s_proj = Z_u.transpose() * p.s;
        const vector y_proj = Z_u.transpose() * p.y;
        detail::bfgs_rank2_update(B, s_proj, y_proj, min_curvature);
    }

    Q_zz = std::move(B);
}

/// Unconstrained legacy apply (Z_u = I, starts from Q_zz as initial B).
inline void lbfgs_apply_to_qzz(matrix &Q_zz,
                                const lbfgs_history &history,
                                scalar_t min_curvature = scalar_t(1e-8)) {
    if (history.empty()) return;
    if (!Q_zz.allFinite()) return;

    const int nu = static_cast<int>(Q_zz.rows());
    matrix B = Q_zz;

    for (const auto &p : history.pairs) {
        if (p.s.size() != nu || p.y.size() != nu) continue;
        detail::bfgs_rank2_update(B, p.s, p.y, min_curvature);
    }

    Q_zz = std::move(B);
}

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_LBFGS_HPP
