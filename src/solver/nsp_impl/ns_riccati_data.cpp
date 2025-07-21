#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
ns_node_data::ns_node_data(const ocp_ptr_t &prob)
    : solver::data_base(prob),
      ns(dense_->prob_->dim(__eq_x)),
      nc(dense_->prob_->dim(__eq_xu)), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr) {
    if (nu < ncstr) {
        nz = 0;
        // throw std::runtime_error("system over-constrained, i.e., nu < ncstr");
    } else {
        nz = nu - ncstr;
    }
    nsp_ = new nullspace_data(dense_->approx_[__eq_x].jac_[__y]);
    nsp_->F_0_k.resize(nx);
    nsp_->F_0_K.resize(nx, nx);
    nsp_->F_u.resize(nx, nu);
    nsp_->z_u_k.resize(nu);
    nsp_->z_u_K.resize(nu, nx);
    nsp_->U_z.resize(nu, nu);
    nsp_->u_z_K.resize(nu, nx);
    nsp_->u_z_k.resize(nu);
    nsp_->s_0_p_k.resize(ns);
    nsp_->s_0_p_K.resize(ns, nx);
    nsp_->s_u.resize(ns, nu);
    nsp_->Q_yy_F_0_K.resize(nx, nx);
    nsp_->s_c_stacked.resize(ncstr, nu);
    nsp_->s_c_stacked_0_k.resize(ncstr);
    nsp_->s_c_stacked_0_K.resize(ncstr, nx);
    // set rollout data for hard constraints
    dual_step[__dyn].resize(nx);
    dual_step[__eq_x].resize(ns);
    dual_step[__eq_xu].resize(nc);
}
ns_node_data::~ns_node_data() {
    delete nsp_;
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto