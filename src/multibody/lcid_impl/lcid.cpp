// #include <moto/multibody/lcid_riccati/lcid_solver.hpp>
// #include <moto/ocp/problem.hpp>
// namespace moto {
// namespace solver {
// namespace lcid_riccati {
// lcid_solver::data::data(ns_riccati_data &d, lcid_solver &solver)
//     : f_x(d.dense_->approx_[__dyn].jac_[__x]),
//       f_u(d.dense_->approx_[__dyn].jac_[__u]) {
//     auto &l = solver.lcid_.as<multibody::lcid>();
//     auto &e = solver.euler_.as<multibody::stacked_euler>();
//     assert(l.nv_ == e.nv_ && "lcid and euler nv mismatch");
//     sp_u_y_K.resize(d.nu, d.nx);
//     d.nz = d.nu - d.ncstr;
//     sp_Z_u.resize(d.nu, d.nz);
//     // setup sparse structures
//     auto &prob = *d.dense_->prob_;
//     size_t dt_st = prob.get_expr_start(e.dt_);
//     size_t a_st = prob.get_expr_start(e.in_args(__u)[0]);
//     sp_u_y_K_a_f.reset(sp_u_y_K.insert(a_st, 0, l.nv_ + l.nc_, d.nx, sparsity::dense));
//     sp_Z_u_a.reset(sp_Z_u.insert(a_st, 0, l.nv_, l.ntq_, sparsity::dense));
//     sp_Z_u_f.reset(sp_Z_u.insert(a_st + l.nv_, 0, l.nc_, l.ntq_, sparsity::dense));
//     sp_Z_u.insert<sparsity::eye>(a_st + l.nv_ + l.nc_, 0, l.ntq_);
//     if (l.has_timestep_)
//         sp_Z_u.insert<sparsity::eye>(dt_st, l.ntq_, 1);
//     if (l.has_timestep_ && d.nz != l.ntq_ + 1) {
//         throw std::runtime_error(fmt::format(
//             "lcid_solver::data: unexpected nz {}, expected {}", d.nz, l.ntq_ + 1));
//     }
//     Q_yy_F_u.resize(d.ny, d.nz);
//     Q_yy_F_u_.reset(Q_yy_F_u.insert(0, 0, d.ny, l.nv_ + l.has_timestep_, sparsity::dense));
//     Q_yy_F_u_Z_u.resize(d.ny, d.nz);
//     F_u_T_Q_yy_F_u_Z_u_.reset(F_u_T_Q_yy_F_u_Z_u.insert(0, 0, l.nv_ + l.has_timestep_, d.nz, sparsity::dense));
//     F_u_T_buf_data_.reset(F_u_T_buf_data.insert(0, 0, l.nv_ + l.has_timestep_, sp_u_y_K.cols(), sparsity::dense));
//     // Q_yy_p.resize(d.nz, d.nz);
//     d.nsp_.Z_y.resize(d.ny, d.nz);
// }
// ns_riccati::ns_riccati_data lcid_solver::create_data(node_data *full_data) {
//     auto d = base::create_data(full_data);
//     d.aux_ = std::make_unique<lcid_solver::data>(d, *this);
//     return d;
// }
// } // namespace lcid_riccati
// } // namespace solver
// } // namespace moto