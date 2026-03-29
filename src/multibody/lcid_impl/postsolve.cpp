// #include <moto/multibody/lcid_riccati/lcid_solver.hpp>
// #include <moto/ocp/impl/node_data.hpp>

// namespace moto {
// namespace solver {
// namespace lcid_riccati {
// void lcid_solver::compute_primal_sensitivity(ns_riccati_data *cur) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
//     d.d_y.k.noalias() = nsp.Z_y * nsp.z_k - nsp.y_y_k;
//     d.d_y.K.noalias() = nsp.Z_y * nsp.z_K - nsp.y_y_K;
//     // nullspace sensitivity is exactly z_K and z_k
// }
// void lcid_solver::compute_primal_sensitivity_correction(ns_riccati_data *cur) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
//     d.d_y.k.noalias() = nsp.Z_y * nsp.z_k - nsp.y_y_k;
// }
// void lcid_solver::finalize_primal_step(ns_riccati_data *cur, bool finalize_dual) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     // recover u from z. U = Zu* z
//     d.trial_prim_step[__u].noalias() = nsp.u_y_k;
//     aux.sp_Z_u.times(aux.z_step, d.trial_prim_step[__u]);
//     aux.sp_u_y_K.times<false>(d.trial_prim_step[__x], d.trial_prim_step[__u]);
//     // multiplier
//     // dynamics multiplier first two terms
//     if (finalize_dual)
//         finalize_dual_newton_step(cur);
// }
// void lcid_solver::finalize_primal_step_correction(ns_riccati_data *cur) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     d.prim_corr[__u].noalias() = nsp.u_y_k;
//     aux.sp_Z_u.times(aux.z_step, d.prim_corr[__u]);
//     aux.sp_u_y_K.times<false>(d.prim_corr[__x], d.prim_corr[__u]);
//     // correction for the primal step
//     for (auto f : primal_fields) {
//         d.trial_prim_step[f] += d.prim_corr[f];
//     }
//     /// update Q_y with correction
//     d.Q_u += d.dense_->merit_jac_modification_[__u];
//     d.Q_y += d.dense_->merit_jac_modification_[__y];
// }
// void lcid_solver::finalize_dual_newton_step(ns_riccati_data *cur) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &l = lcid_.as<multibody::lcid>();
//     auto &ld = d.full_data_->data(lcid_).as<multibody::lcid::data>();
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     d.d_lbd_f.noalias() = -d.Q_y.transpose() - d.V_yy * d.trial_prim_step[__y];
//     d.Q_yx.times<false>(d.trial_prim_step[__x], d.d_lbd_f);
//     d.Q_yx_mod.times<false>(d.trial_prim_step[__x], d.d_lbd_f);
//     // update hard constraint multipliers
//     // LU.solve([rhs])
//     d.d_lbd_s_c_pre_solve.noalias() = -d.Q_u.transpose();
//     d.Q_ux.times<false>(d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve);
//     d.Q_ux_mod.times<false>(d.trial_prim_step[__x], d.d_lbd_s_c_pre_solve);
//     d.Q_uu.times<false>(d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve);
//     d.Q_uu_mod.times<false>(d.trial_prim_step[__u], d.d_lbd_s_c_pre_solve);
//     d.F_u.T_times<false>(d.d_lbd_f, d.d_lbd_s_c_pre_solve);
//     // d.d_lbd_s_c.noalias() = nsp.lu_eq_.transpose().solve(d.d_lbd_s_c_pre_solve);
//     d.d_lbd_s_c.noalias() = ld.G_ * d.d_lbd_s_c_pre_solve;
//     d.trial_dual_step[__eq_xu] = d.d_lbd_s_c.tail(d.nc);
//     // d.trial_dual_step[__dyn].noalias() =  nsp.lu_dyn_.transpose().solve(d.d_lbd_f);
//     cur->apply_jac_y_inverse_transpose(d.d_lbd_f, d.trial_dual_step[__dyn]);
// }
// } // namespace lcid_riccati
// } // namespace solver
// } // namespace moto