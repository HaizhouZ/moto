#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>

#include <moto/utils/field_conversion.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void generic_solver::fwd_linear_rollout(ns_riccati_data *cur, ns_riccati_data *next) {
    auto &d = *cur;
    d.trial_prim_step[__y].noalias() = d.d_y.k + d.d_y.K * d.trial_prim_step[__x];
    if (next != nullptr) [[likely]] {
        utils::copy_y_to_x_tangent(d.trial_prim_step[__y], next->trial_prim_step[__x], cur->dense_->prob_, next->dense_->prob_);
    }
}
void generic_solver::finalize_dual_newton_step(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.V_yy * d.trial_prim_step[__y];
    d.Q_yx.times<false>(d.trial_prim_step[__x], d.d_lbd_f);
    d.Q_yx_mod.times<false>(d.trial_prim_step[__x], d.d_lbd_f);
    // update hard constraint multipliers
    if (d.ncstr > 0 && d.rank_status_ != rank_status::unconstrained) {
        // LU.solve([rhs])
        d.d_lbd_s_c_pre_solve.noalias() = -d.Q_u.transpose();
        d.Q_ux.times<false>(d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve);
        d.Q_ux_mod.times<false>(d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve);
        d.Q_uu.times<false>(d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve);
        d.Q_uu_mod.times<false>(d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve);
        d.F_u.T_times<false>(d.d_lbd_f, d.d_lbd_s_c_pre_solve);
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
            d.s_y.T_times<false>(d.trial_dual_step[__eq_x], d.d_lbd_f);
        }
        if (d.nc > 0) {
            d.trial_dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
        }
    } else if (d.ncstr > 0 && d.aux_ != nullptr) {
        // GN mode: mirrors pmm_constr::finalize_newton_step — dlam = (h + J*du) / rho_eq
        const scalar_t rho_eq = static_cast<ns_riccati_data::restoration_aux_data *>(d.aux_.get())->rho_eq;
        d.d_lbd_s_c.noalias() = nsp.s_c_stacked_0_k;                           // h
        d.d_lbd_s_c.noalias() += nsp.s_c_stacked * d.trial_prim_step[__u];     // + J_u * du
        d.d_lbd_s_c.noalias() += nsp.s_c_stacked_0_K * d.trial_prim_step[__x]; // + J_x * dx
        d.d_lbd_s_c *= scalar_t(1) / rho_eq;                                   // / rho_eq
        if (d.ns > 0) {
            d.trial_dual_step[__eq_x] = d.d_lbd_s_c.head(d.ns);
            d.s_y.T_times<false>(d.trial_dual_step[__eq_x], d.d_lbd_f);
        }
        if (d.nc > 0) {
            d.trial_dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
        }
    }
    cur->apply_jac_y_inverse_transpose(d.d_lbd_f, d.trial_dual_step[__dyn]);
}
void generic_solver::finalize_primal_step(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    d.trial_prim_step[__u].noalias() = d.d_u.k + d.d_u.K * d.trial_prim_step[__x];
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
    d.prim_corr[__u].noalias() = d.d_u.k + d.d_u.K * d.prim_corr[__x];
    // correction for the primal step
    for (auto f : primal_fields) {
        d.trial_prim_step[f] += d.prim_corr[f];
    }
    /// update Q_y with correction
    d.Q_u += d.dense_->merit_jac_modification_[__u];
    d.Q_y += d.dense_->merit_jac_modification_[__y];
}
void generic_solver::compute_kkt_residual(ns_riccati_data *cur) {
    auto &d = *cur;
    // compute KKT residual
    auto dense = d.dense_;
    auto &f_u = dense->approx_[__dyn].jac_[__u];
    dense->res_stat_[__u].noalias() = d.Q_u_bak.transpose(); // d.d_lbd_f
    d.Q_uu.times(d.trial_prim_step[__u], dense->res_stat_[__u]);
    dense->res_stat_[__y].noalias() = d.Q_y_bak.transpose();
    d.Q_yy.times(d.trial_prim_step[__y], dense->res_stat_[__y]);
    dense->res_stat_[__x].noalias() = d.Q_x_bak.transpose();
    d.Q_xx.times(d.trial_prim_step[__x], dense->res_stat_[__x]);
    for (auto f : primal_fields) {
        for (auto constr : constr_fields) {
            if (dense->approx_[constr].jac_[f].is_empty() || d.trial_dual_step[constr].size() == 0) {
                continue;
            }
            dense->approx_[constr].jac_[f].right_T_times(d.trial_dual_step[constr], dense->res_stat_[f]);
        }
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
