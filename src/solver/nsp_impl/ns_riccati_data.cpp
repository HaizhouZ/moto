#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
ns_node_data::ns_node_data(sym_data *s, dense_approx_data *dense)
    : solver::data_base(s, dense),
      ns(dense->approx_[__eq_x].v_.size()),
      nc(dense->approx_[__eq_xu].v_.size()), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx), d_lbd_s_c_pre_solve(nu), d_lbd_s_c(ncstr),
      nsp_(new nullspace_data(dense->approx_[__eq_x].jac_[__y])) {
    if (nu < ncstr) {
        nz = 0;
        // throw std::runtime_error("system over-constrained, i.e., nu < ncstr");
    } else {
        nz = nu - ncstr;
    }
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
    if (nsp_)
        delete nsp_.get();
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto