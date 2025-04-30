#include <atri/solver/ns_riccati_data.hpp>

namespace atri {

nullspace_riccati_data::nullspace_riccati_data(problem_ptr_t exprs)
    : node_data(exprs), nx(exprs->dim_[__x]), nu(exprs->dim_[__u]),
      ns(raw_data_.exprs_->dim_[__eq_cstr_s]),
      nc(raw_data_.exprs_->dim_[__eq_cstr_c]), ncstr(ns + nc), d_u(nu, nx),
      d_y(nx, nx), d_lbd_f(nx, nx), d_lbd_s_c(ncstr, nx), F_u(nx, nu),
      Q_x(raw_data_.jac_[__x]), Q_u(raw_data_.jac_[__u]),
      Q_y(raw_data_.jac_[__y]),
      Q_xx(raw_data_.hessian_[__x][__x]),
      Q_xu(raw_data_.hessian_[__x][__u]),
      Q_uu(raw_data_.hessian_[__u][__u]),
      Q_xy(raw_data_.hessian_[__x][__y]),
      Q_yy(raw_data_.hessian_[__y][__y]),
      s_y(raw_data_.approx_[__eq_cstr_s].jac_[__y]) {
    nz = nu - ncstr;
    z_u_k.resize(nu);
    z_u_K.resize(nu, nx);
    U_z.resize(nz, nz);
    s_0_p_k.resize(ns);
    s_0_p_K.resize(ns, nx);
    s_u.resize(ns, nu);
    s_c_stacked.resize(ncstr, nu);
    s_c_stacked_0_k.resize(ncstr);
    s_c_stacked_0_K.resize(ncstr, nx);
}
} // namespace atri