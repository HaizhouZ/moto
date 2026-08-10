#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/core/linear_backend.hpp>

#include <moto/utils/field_conversion.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
namespace {
void recover_input_sensitivity(ns_riccati_data &d, bool correction) {
    auto &nsp = d.nsp_;
    if (d.rank_status_ == rank_status::unconstrained) {
        d.d_u.k = nsp.z_k;
        d.d_u.K = nsp.z_K;
    } else if (d.rank_status_ == rank_status::fully_constrained) {
        if (correction)
            d.d_u.k.setZero();
        else
            d.d_u.k = -nsp.u_y_k;
        d.d_u.K = -nsp.u_y_K;
    } else {
        d.d_u.k.noalias() = nsp.Z_u * nsp.z_k;
        if (!correction)
            d.d_u.k -= nsp.u_y_k;
        d.d_u.K.noalias() = nsp.Z_u * nsp.z_K - nsp.u_y_K;
    }
}

void recover_lifted_direction(ns_riccati_data &d, const vector &dx,
                              const vector &du, vector &dl,
                              bool correction) {
    dl.setZero();
    linear_backend::multiply(d.lifting_.l_x(), dx, dl, -1.);
    linear_backend::multiply(d.lifting_.l_u(), du, dl, -1.);
    if (!correction)
        dl -= d.lifting_.l_0();
}
} // namespace

void generic_solver::fwd_linear_rollout(ns_riccati_data *cur, ns_riccati_data *next) {
    auto &d = *cur;
    d.trial_prim_step[__y].noalias() =
        d.d_y.k + d.d_y.K * d.trial_prim_step[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x_tangent(d.trial_prim_step[__y], next->trial_prim_step[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void generic_solver::finalize_dual_newton_step(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    d.d_lbd_f.noalias() =
        -d.Q_y.transpose() - d.V_yy * d.trial_prim_step[__y];
    linear_backend::multiply(d.Q_yx, d.trial_prim_step[__x],
                             d.d_lbd_f, -1.);
    linear_backend::multiply(d.Q_yx_mod, d.trial_prim_step[__x],
                             d.d_lbd_f, -1.);
    // update hard constraint multipliers
    if (d.ncstr > 0 && d.rank_status_ != rank_status::unconstrained) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -d.Q_u.transpose();
        linear_backend::multiply(d.Q_ux, d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve, -1.);
        linear_backend::multiply(d.Q_ux_mod, d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve, -1.);
        linear_backend::multiply(d.Q_uu, d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve, -1.);
        linear_backend::multiply(d.Q_uu_mod, d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve, -1.);
        linear_backend::transpose_multiply(d.F_u, d.d_lbd_f, d.d_lbd_s_c_pre_solve, -1.);
        // solve for hard constraint multiplers
        // fmt::print("Q_y: \n{}\n", d.Q_y);
        // fmt::print("Q_x: \n{}\n", d.Q_x);
        // fmt::print("Q_zz: \n{}\n", nsp.Q_zz);
        // fmt::print("Q_zz eigenvalues: {}\n", nsp.Q_zz.eigenvalues().transpose());
        // fmt::print("\n");
        // fmt::print("Q_u: \n{}\n", d.Q_u);
        // fmt::print("Q_yy: \n{}\n", d.Q_yy);
        // fmt::print("V_xx: \n{}\n", d.V_xx);
        // fmt::print("V_yy: \n{}\n", d.V_yy);
        // fmt::print("Z_u: \n{}\n", nsp.Z_u);
        // fmt::print("Z_y: \n{}\n", nsp.Z_y);
        // fmt::print("Z_k: \n{}\n", nsp.z_K);
        // fmt::print("u_y_k: \n{}\n", nsp.u_y_k.transpose());
        // fmt::print("y_y_k: \n{}\n", nsp.y_y_k.transpose());
        // fmt::print("y_y_K: \n{}\n", nsp.y_y_K);
        // fmt::print("y_0_p_k: \n{}\n", nsp.y_0_p_k.transpose());
        // fmt::print("y_0_p_K: \n{}\n", nsp.y_0_p_K);
        // fmt::print("d_lbd_f: \n{}\n", d.d_lbd_f.transpose());
        // fmt::print("d_lbd_s_c_pre_solve: \n{}\n", d.d_lbd_s_c_pre_solve.transpose());
        d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
        // fmt::print("pre solve hard constr multipliers: {}\n", d.d_lbd_s_c_pre_solve.transpose());

        size_t cur_idx = 0;
        if (d.ns > 0) {
            // append last term in dynamics multipler computation
            d.trial_dual_step[__eq_x] = d.d_lbd_s_c.head(d.ns);
            cur_idx += d.ns;
            linear_backend::transpose_multiply(d.s_y, d.trial_dual_step[__eq_x], d.d_lbd_f, -1.);
        }
        if (d.nc > 0) {
            d.trial_dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
        }
    }
    cur->recover_lifted_dual(d.d_lbd_f);
}
void generic_solver::finalize_primal_step(ns_riccati_data *cur) {
    auto &d = *cur;
    recover_input_sensitivity(d, false);
    d.trial_prim_step[__u].noalias() =
        d.d_u.k + d.d_u.K * d.trial_prim_step[__x];
    recover_lifted_direction(d, d.trial_prim_step[__x],
                             d.trial_prim_step[__u],
                             d.trial_prim_step[__l], false);
}
void generic_solver::fwd_linear_rollout_correction(ns_riccati_data *cur, ns_riccati_data *next) {
    auto &d = *cur;
    d.prim_corr[__y].noalias() = d.d_y.k + d.d_y.K * d.prim_corr[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x_tangent(d.prim_corr[__y], next->prim_corr[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void generic_solver::finalize_primal_step_correction(ns_riccati_data *cur) {
    auto &d = *cur;
    recover_input_sensitivity(d, true);
    d.prim_corr[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_corr[__x];
    recover_lifted_direction(d, d.prim_corr[__x], d.prim_corr[__u],
                             d.prim_corr[__l], true);
    // correction for the primal step
    for (auto f : primal_fields) {
        d.trial_prim_step[f] += d.prim_corr[f];
    }
    d.Q_u += d.dense_->lag_jac_corr_[__u];
    d.Q_y += d.dense_->lag_jac_corr_[__y];
    d.Q_l += d.dense_->lag_jac_corr_[__l];
}
void generic_solver::compute_kkt_residual(ns_riccati_data *cur) {
    auto &d = *cur;
    auto dense = d.dense_;
    for (const auto field : primal_fields)
        d.kkt_stat_err_[field] = d.base_lag_grad_backup[field].transpose();

    // Apply the complete full-space symmetric Hessian. The NSP stores only
    // lower block-triangular panels, so every off-diagonal block contributes
    // to both stationarity rows.
    for (size_t i = 0; i < primal_fields.size(); ++i) {
        const auto fi = primal_fields[i];
        for (size_t j = 0; j <= i; ++j) {
            const auto fj = primal_fields[j];
            const auto &h = dense->lag_hess_[fi][fj];
            if (!h.is_empty()) {
                linear_backend::multiply(h, d.trial_prim_step[fj],
                                         d.kkt_stat_err_[fi]);
                if (i != j)
                    linear_backend::right_transpose_multiply(
                        d.trial_prim_step[fi], h,
                        d.kkt_stat_err_[fj]);
            }
        }
    }

    for (auto f : primal_fields) {
        for (auto constr : constr_fields) {
            if (dense->approx_[constr].jac_[f].is_empty() || d.trial_dual_step[constr].size() == 0) {
                continue;
            }
            linear_backend::right_transpose_multiply(d.trial_dual_step[constr], dense->approx_[constr].jac_[f], d.kkt_stat_err_[f]);
        }
    }
    if (std::getenv("MOTO_DEBUG_LIFTING_DUAL") && d.nl && d.nl <= 4) {
        fmt::println("lifted KKT u={} l={} dlift={} Jlift_u={}",
                     d.kkt_stat_err_[__u].transpose(),
                     d.kkt_stat_err_[__l].transpose(),
                     d.trial_dual_step[__lift].transpose(),
                     dense->approx_[__lift].jac_[__u].dense());
    }
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto
