#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/data/rollout_data.hpp>
#include <atri/solver/ns_riccati_data.hpp>
#include <atri/ocp/core/problem_data.hpp>

namespace atri {

nullspace_riccati_data::nullspace_riccati_data(problem_ptr_t prob)
    : node_data(prob), nx(prob->dim_[__x]), nu(prob->dim_[__u]),
      ns(raw_->prob_->dim_[__eq_cstr_s]),
      nc(raw_->prob_->dim_[__eq_cstr_c]), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx, nx), d_lbd_s_c(ncstr, nx),
      Q_x(raw_->jac_[__x]), Q_u(raw_->jac_[__u]),
      Q_y(raw_->jac_[__y]), Q_xx(raw_->hessian_[__x][__x]),
      Q_xu(raw_->hessian_[__x][__u]), Q_uu(raw_->hessian_[__u][__u]),
      Q_xy(raw_->hessian_[__x][__y]), Q_yy(raw_->hessian_[__y][__y]) {
    nz = nu - ncstr;
    nsp_ = new nullspace_data(raw_->approx_[__eq_cstr_s].jac_[__y]);
    nsp_->F_u.resize(nx, nu);
    nsp_->z_u_k.resize(nu);
    nsp_->z_u_K.resize(nu, nx);
    nsp_->U_z.resize(nz, nz);
    nsp_->s_0_p_k.resize(ns);
    nsp_->s_0_p_K.resize(ns, nx);
    nsp_->s_u.resize(ns, nu);
    nsp_->s_c_stacked.resize(ncstr, nu);
    nsp_->s_c_stacked_0_k.resize(ncstr);
    nsp_->s_c_stacked_0_K.resize(ncstr, nx);
    rollout_ = new rollout_data();
    rollout_->prim_[__x].resize(nx);
    rollout_->prim_[__u].resize(nu);
    rollout_->prim_[__y].resize(nx);
    rollout_->dual_[__dyn].resize(nx);
    rollout_->dual_[__eq_cstr_s].resize(ns);
    rollout_->dual_[__eq_cstr_c].resize(nc);
}
nullspace_riccati_data::~nullspace_riccati_data() {
    delete nsp_;
    delete rollout_;
}
} // namespace atri