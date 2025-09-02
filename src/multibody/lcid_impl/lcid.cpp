#include <moto/multibody/lcid_riccati/lcid_solver.hpp>

namespace moto {
namespace solver {
namespace lcid_riccati {
lcid_solver::data::data(ns_riccati_data &d, lcid_solver &solver)
    : f_x(d.dense_->approx_[__dyn].jac_[__x]),
      f_u(d.dense_->approx_[__dyn].jac_[__u]),
      f_y(d.dense_->approx_[__dyn].jac_[__y]) {
    auto &l = solver.lcid_.as<multibody::lcid>();
    auto &e = solver.euler_.as<multibody::stacked_euler>();
    assert(l.nv_ == e.nv_ && "lcid and euler nv mismatch");
    sp_u_y_K.resize(d.nu, d.nx);
    sp_u_y_k.resize(d.nu);
    sp_y_y_k.resize(d.ny);
    nz = d.nu - d.ncstr;
    sp_Z_u.resize(d.nu, nz);
    f_z.resize(d.ny, nz);
    f_z_T_Q_yy_p.resize(nz, d.ny);
    // setup sparse structures
    sp_u_y_K_a_f.reset(sp_u_y_K.insert(0, 0, l.nv_ + l.nc_, d.nx, sparsity::dense));
    sp_Z_u_a.reset(sp_Z_u.insert(0, 0, l.nv_, l.ntq_, sparsity::dense));
    sp_Z_u_f.reset(sp_Z_u.insert(l.nv_, 0, l.nc_, l.ntq_, sparsity::dense));
    sp_Z_u.insert<sparsity::eye>(l.nv_ + l.nc_, 0, nz);
    if (l.has_timestep_ && nz != l.ntq_ + 1) {
        throw std::runtime_error(fmt::format(
            "lcid_solver::data: unexpected nz {}, expected {}", nz, l.ntq_ + 1));
    }
    f_z_a.reset(f_z.insert(l.nv_, 0, l.nv_, l.nv_, sparsity::dense));
    if (l.has_timestep_) {
        f_z_dt.reset(f_z.insert(0, l.nv_, l.nv_, 1, sparsity::dense));
    }
    f_a_times_u_y_K.reset(f_u_times_u_y_K.insert(l.nv_, 0, l.nv_, l.nv_, sparsity::dense));
}
auto lcid_solver::create_data(node_data *full_data) -> ns_riccati_data * {
    auto d = base::create_data(full_data);
    d->aux_ = std::make_unique<lcid_solver::data>(*d, *this);
    return d;
}
} // namespace lcid_riccati
} // namespace solver
} // namespace moto