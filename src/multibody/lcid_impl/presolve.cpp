#include <moto/multibody/lcid_riccati/lcid_solver.hpp>
#include <moto/ocp/impl/node_data.hpp>
namespace moto {
namespace solver {
namespace lcid_riccati {
void lcid_solver::ns_factorization(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
    auto &full_data = *d.full_data_;
    auto &l = lcid_.as<multibody::lcid>();
    auto &ld = full_data.data(lcid_).as<multibody::lcid::data>();
    auto &nsp = d.nsp_;
    auto &c_x = d.dense_->approx_[__eq_xu].jac_[__x];
    // prepare
    nsp.u_0_p_k = d.Q_u.transpose();
    nsp.u_0_p_K.setZero();
    d.Q_ux.dump_into(nsp.u_0_p_K);
    d.Q_ux_mod.dump_into(nsp.u_0_p_K);
    nsp.y_0_p_k.setZero();
    d.V_xx.setZero();
    d.Q_xx.dump_into(d.V_xx);
    d.Q_xx_mod.dump_into(d.V_xx);
    d.V_yy.setZero();
    d.Q_yy.dump_into(d.V_yy);
    d.Q_yy_mod.dump_into(d.V_yy);
    // setup M matrix of l
    for (size_t id = 0; id < l.dyn_constr_.size(); ++id) {
        const auto &con = l.dyn_constr_[id];
        auto &s = con.s[0];
        auto &dyn_data = full_data.data(con.c);
        auto &M = ld.Minv_.dense_panels_[id].data_;
        M = dyn_data.jac(s.a);
    }
    // setup jacs
    size_t ik_idx = 0;
    for (auto &con : l.kin_constr_) {
        auto &kin_data = full_data.data(con.c);
        for (auto &s : con.s) {
            auto &J_T = ld.Jc_T_.dense_panels_[ik_idx++].data_;
            J_T = kin_data.jac(s.a).transpose();
        }
    }
    // compute inverses
    l.compute_osim_inv(ld);
    // compute nullspace projector
    aux.sp_Z_u_a.setZero();
    ld.tq_S_.right_times<false>(ld.G_a_, aux.sp_Z_u_a.get());
    aux.sp_Z_u_f.setZero();
    ld.tq_S_.right_times<false>(ld.G_off_diag_, aux.sp_Z_u_f.get());
    // compute pseudo sol
    c_x.right_times(ld.G_, aux.sp_u_y_K_a_f.get());
    // compute fz
    auto &e = euler_.as<multibody::stacked_euler>();
    auto &ed = full_data.data(euler_).as<multibody::stacked_euler::approx_data>();
    aux.f_z_a.noalias() = ed.f_u_v_diag_.asDiagonal() * aux.sp_Z_u_a;
    if (l.has_timestep_) {
        aux.f_z_dt.array() = ed.f_t.array();
    }
    // compute Q_zz U part
    thread_local moto::utils::buffer_tpl<matrix> buf;
    // nsp.Q_zz.noalias() = nsp.Z_u.transpose() * d.Q_uu * nsp.Z_u;
    buf.resize(d.nu, nsp.Z_u.cols());
    buf.data_.setZero();
    d.Q_uu.times(aux.sp_Z_u, buf.data_);
    d.Q_uu_mod.times(aux.sp_Z_u, buf.data_);
    nsp.Q_zz.setZero();
    aux.sp_Z_u.T_times(buf.data_, nsp.Q_zz);
    // update Z_0_K
    d.Q_uu.times<false>(aux.sp_u_y_K, nsp.u_0_p_K);
    d.Q_uu_mod.times<false>(aux.sp_u_y_K, nsp.u_0_p_K);
    nsp.z_0_K.setZero();
    aux.sp_Z_u.T_times(nsp.u_0_p_K, nsp.z_0_K);
    aux.sp_Z_u.T_times(d.Q_ux, nsp.z_0_K);
    aux.sp_Z_u.T_times(d.Q_ux_mod, nsp.z_0_K);
    nsp.y_0_p_K.setZero();
    // for now assume Q_yx is zero
    // aux.f_z.T_times(d.Q_yx, nsp.z_0_K); // Z_0_K = f_z^T * Q_yx + ,,,
    aux.f_a_times_u_y_K.noalias() = ed.f_u_v_diag_.asDiagonal() * aux.sp_u_y_K_a_f.topRows(l.nv_);
    // update zero order terms
    nsp.u_y_k.setZero();
    nsp.u_y_k.head(l.nv_ + l.nc_).noalias() = ld.G_ * d.dense_->approx_[__eq_xu].v_;
    d.F_u.times(aux.sp_Z_u, nsp.Z_y);
    nsp.y_y_k = d.dense_->approx_[__dyn].v_;
    aux.f_u.times<false>(nsp.u_y_k, nsp.y_y_k);
    nsp.z_0_k.setZero();
    d.Q_uu.times<false>(nsp.u_y_k, nsp.u_0_p_k);
    d.Q_uu_mod.times<false>(nsp.u_y_k, nsp.u_0_p_k);
    aux.sp_Z_u.T_times(nsp.u_0_p_k, nsp.z_0_k);
    // precompute value function terms
    aux.sp_u_y_K.right_T_times<false>(nsp.u_0_p_k, d.Q_x);
    aux.sp_u_y_K.right_T_times<false>(nsp.u_0_p_K, d.V_xx);
    d.Q_ux.right_T_times<false>(nsp.u_y_k, d.Q_x);
    d.Q_ux_mod.right_T_times<false>(nsp.u_y_k, d.Q_x);
    // d.Q_yx.right_T_times<false>(nsp.y_y_k, d.Q_x);
    // d.Q_yx_mod.right_T_times<false>(nsp.y_y_k, d.Q_x);
    d.Q_ux.right_T_times<false>(aux.sp_u_y_K, d.V_xx);
    d.Q_ux_mod.right_T_times<false>(aux.sp_u_y_K, d.V_xx);
    // d.Q_yx.right_T_times<false>(nsp.y_y_K, d.V_xx);
    // d.Q_yx_mod.right_T_times<false>(nsp.y_y_K, d.V_xx);
}
} // namespace lcid_riccati
} // namespace solver
} // namespace moto