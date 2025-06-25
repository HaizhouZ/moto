#include <moto/ocp/approx_storage.hpp>
#include <moto/solver/data/nullspace_data.hpp>
#include <moto/solver/data/rollout_data.hpp>
#include <moto/solver/ns_riccati_data.hpp>

namespace moto {
namespace ns_riccati {
riccati_data::riccati_data(const ocp_ptr_t& prob)
    : node_data(prob), nx(prob->dim_[__x]), nu(prob->dim_[__u]),
      ns(dense_->prob_->dim_[__eq_cstr_s]),
      nc(dense_->prob_->dim_[__eq_cstr_c]), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr),
      Q_x(dense_->jac_[__x]), Q_u(dense_->jac_[__u]),
      Q_y(dense_->jac_[__y]), Q_xx(dense_->hessian_[__x][__x]),
      Q_ux(dense_->hessian_[__u][__x]), Q_uu(dense_->hessian_[__u][__u]),
      Q_yx(dense_->hessian_[__y][__x]), Q_yy(dense_->hessian_[__y][__y]) {
    nz = nu - ncstr;
    nsp_ = new nullspace_data(dense_->approx_[__eq_cstr_s].jac_[__y]);
    nsp_->F_0_k.resize(nx);
    nsp_->F_0_K.resize(nx, nx);
    nsp_->F_u.resize(nx, nu);
    nsp_->z_u_k.resize(nu);
    nsp_->z_u_K.resize(nu, nx);
    nsp_->U_z.resize(nz, nz);
    nsp_->s_0_p_k.resize(ns);
    nsp_->s_0_p_K.resize(ns, nx);
    nsp_->s_u.resize(ns, nu);
    nsp_->Q_yy_F_0_K.resize(nx, nx);
    nsp_->s_c_stacked.resize(ncstr, nu);
    nsp_->s_c_stacked_0_k.resize(ncstr);
    nsp_->s_c_stacked_0_K.resize(ncstr, nx);
    rollout_ = new rollout_data();
    rollout_->prim_[__x].resize(nx);
    rollout_->prim_[__x].setZero();
    rollout_->prim_[__u].resize(nu);
    rollout_->prim_[__y].resize(nx);
    rollout_->dual_[__dyn].resize(nx);
    rollout_->dual_[__eq_cstr_s].resize(ns);
    rollout_->dual_[__eq_cstr_c].resize(nc);
}
riccati_data::~riccati_data() {
    delete nsp_;
    delete rollout_;
}
} // namespace ns_riccati
} // namespace moto