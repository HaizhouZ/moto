#include <moto/ocp/approx_storage.hpp>
#include <moto/solver/ns_riccati_data.hpp>
#include <moto/solver/nullspace_data.hpp>

namespace moto {
namespace ns_riccati {
riccati_data::riccati_data(const ocp_ptr_t &prob)
    : node_data(prob), solver::solver_data(prob, dense_.get()),
      ns(dense_->prob_->dim_[__eq_x]),
      nc(dense_->prob_->dim_[__eq_xu]), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr) {
    nz = nu - ncstr;
    nsp_ = new nullspace_data(dense_->approx_[__eq_x].jac_[__y]);
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
    // set rollout data for hard constraints
    dual_rollout_[__dyn].resize(nx);
    dual_rollout_[__eq_x].resize(ns);
    dual_rollout_[__eq_xu].resize(nc);
}
riccati_data::~riccati_data() {
    delete nsp_;
}
} // namespace ns_riccati
} // namespace moto