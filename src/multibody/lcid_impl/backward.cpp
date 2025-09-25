// #include <moto/multibody/lcid_riccati/lcid_solver.hpp>
// #include <moto/ocp/impl/node_data.hpp>
// #include <moto/utils/field_conversion.hpp>
// // #define ENABLE_TIMED_BLOCK
// #include <moto/utils/timed_block.hpp>
// namespace moto {
// namespace solver {
// namespace lcid_riccati {
// void lcid_solver::riccati_recursion(ns_riccati_data *cur, ns_riccati_data *prev) {
//     auto &d = *cur;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     auto &l = lcid_.as<multibody::lcid>();
//     auto &nsp = d.nsp_;
//     auto &ld = d.full_data_->data(lcid_).as<multibody::lcid::data>();
//     auto &e = euler_.as<multibody::stacked_euler>();
//     auto &ed = d.full_data_->data(euler_).as<multibody::stacked_euler::approx_data>();
//     d.V_yy.array() /= 2;
//     /// @todo: temporary
//     d.V_yy = d.V_yy + d.V_yy.transpose().eval();
//     // aux.f_y_inv->inner_product(d.V_yy, d.V_yy);
//     // update Q_zz
//     // d.F_u.right_times(d.V_yy, aux.Q_yy_F_u_.get());
//     // // fmt::println("Q_yy_F_u_:{}\n", aux.Q_yy_F_u_.get());
//     // aux.sp_Z_u.right_times(aux.Q_yy_F_u, aux.Q_yy_F_u_Z_u);
//     // // fmt::println("Q_yy_F_u_Z_u:{}", Q_yy_F_u_Z_u);
//     // d.F_u.T_times(aux.Q_yy_F_u_Z_u, aux.F_u_T_Q_yy_F_u_Z_u_.get());
//     // // fmt::println("F_u_T_Q_yy_F_u_Z_u:{}", F_u_T_Q_yy_F_u_Z_u_.get());
//     // aux.sp_Z_u.T_times(aux.F_u_T_Q_yy_F_u_Z_u, nsp.Q_zz);
//     // timed_block_end("step1: update Q_zz");
//     // // fmt::println("Q_zz:{}", nsp.Q_zz);
//     // // update z_0_k += f_z^T ( f_y_inv.T * Q_y.T - V_yy_p * y_y_k)
//     nsp.y_0_p_k.noalias() = d.Q_y.transpose() - d.V_yy * nsp.y_y_k;
//     nsp.z_0_k.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_k;
//     // // update z_0_K += f_z^T * V_yy_p * y_y_K (i.e., f_x - f_u * u_y_K)
//     // timed_block_start("step2: update z_0_K");
//     // d.F_x.right_T_times(aux.Q_yy_F_u_Z_u, nsp.z_0_K);
//     // aux.sp_u_y_K.right_T_times<false>(aux.F_u_T_Q_yy_F_u_Z_u, nsp.z_0_K);
    
//     // y_0_p_K.T * nsp,y_y_K = (-y_y_K.T * V_yy * y_y_K)
//     thread_local utils::buffer_tpl<matrix> buf;
//     // buf.resize(d.V_yy.rows(), nsp.y_y_K.cols());
//     // buf.data_.setZero();
//     timed_block_start("step3: update Vx, Qx");
//     // timed_block_start("step3.1: F_x.right_times");
//     // d.F_x.right_times(d.V_yy, buf.data_);
//     // timed_block_end("step3.1: F_x.right_times");
//     // timed_block_start("step3.2: sp_u_y_K.right_times");
//     // aux.sp_u_y_K.right_times(aux.Q_yy_F_u, buf.data_);
//     // // fmt::println("Q_yy_F_u:\n{:.2}", aux.Q_yy_F_u.dense());
//     // timed_block_end("step3.2: sp_u_y_K.right_times");

//     nsp.y_0_p_K.noalias() = d.V_yy * nsp.y_y_K;

//     timed_block_start("step1: update Q_zz");
//     nsp.z_0_K.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_K;
//     nsp.Q_zz.noalias() += nsp.Z_y.transpose() * d.V_yy * nsp.Z_y;
//     timed_block_end("step2: update z_0_K");
//     timed_block_start("step2: solve Qzz");
//     nsp.llt_ns_.compute(nsp.Q_zz);
//     nsp.llt_ns_.solve(nsp.z_0_K, nsp.z_K, -1.0);
//     timed_block_end("step2: solve Qzz");

    
//     // timed_block_start("step3.3: z_0_K.T * buf");
//     // d.V_xx.noalias() += buf.data_.transpose() * nsp.y_y_K;
//     // timed_block_end("step3.3: z_0_K.T * buf");
//     // update value function derivatives
//     timed_block_start("step3: update Vx, Qx 1/2");
//     d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K - nsp.y_0_p_k.transpose() * nsp.y_y_K;
//     d.V_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K - nsp.y_0_p_K.transpose() * nsp.y_y_K;
//     timed_block_end("step3: update Vx, Qx 1/2");
    
//     // timed_block_start("step3.3: F_x.T_times");
//     // d.F_x.T_times(buf.data_, d.V_xx);
//     // // fmt::println("F_x:\n{:.2}", d.F_x.dense());
//     // // getchar();
//     // timed_block_end("step3.3: F_x.T_times");

//     // timed_block_start("step3.4: F_u.T_times");
//     // d.F_u.T_times(buf.data_, aux.F_u_T_buf_data_.get());
//     // // fmt::println("F_u:\n{:.2}", d.F_u.dense());
//     // // getchar();
//     // timed_block_end("step3.4: F_u.T_times");

//     // timed_block_start("step3.5: sp_u_y_K.T_times");
//     // aux.sp_u_y_K.T_times(aux.F_u_T_buf_data, d.V_xx);
//     // // fmt::println("sp_u_y_K:\n{:.2}", aux.sp_u_y_K.dense());
//     // // getchar();
//     // matrix V_xx_(d.V_xx.rows(), d.V_xx.cols());
//     // aux.sp_u_y_K.T_times(aux.F_u_T_buf_data, V_xx_);
//     // // fmt::println("F_u_T_buf_data:\n{:.2}", aux.F_u_T_buf_data.dense());
//     // fmt::println("y_y_K:\n{:.2}", nsp.y_y_K);
//     // timed_block_end("step3.5: sp_u_y_K.T_times");
//     // timed_block_end("step3: update Vx, Qx");
//     // update value function derivatives of previous node
//     if (prev != nullptr) [[likely]] {
//         auto &d_pre = *prev;
//         auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
//         d.Q_x *= perm;  
//         d.V_xx *= perm;
//         d.V_xx.applyOnTheLeft(perm.transpose());
//         d_pre.Q_y.noalias() += d.Q_x;
//         d_pre.V_yy.noalias() += d.V_xx;
//     }
// }
// void lcid_solver::riccati_recursion_correction(ns_riccati_data *cur, ns_riccati_data *prev) {
//     auto &d = *cur;
//     auto &nsp = d.nsp_;
//     auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
//     // Z_y^T = Z_u * f_u^T * f_y^{-T} = f_z^T * f_y^{-T}
//     nsp.z_0_k.noalias() = nsp.Z_y.transpose() * d.Q_y.transpose();
//     aux.sp_Z_u.T_times(d.Q_u, nsp.z_0_k);
//     // d.Q_x.noalias() = nsp.z_0_k.transpose() * nsp.z_K - (d.Q_u * nsp.u_y_K + d.Q_y * nsp.y_y_K);
//     d.Q_x.noalias() = nsp.z_0_k.transpose() * nsp.z_K - (d.Q_y * nsp.y_y_K);
//     aux.sp_u_y_K.right_times<false>(d.Q_u, d.Q_x);
//     // compute Q_x correcton
//     // d.Q_x.noalias() = -d.Q_y * nsp.F_0_K + nsp.z_u_k.transpose() * d.d_u.K;
//     // d.Q_x.noalias() = -d.Q_y * d.F_x.dense() + nsp.z_u_k.transpose() * d.d_u.K;
//     if (prev != nullptr) [[likely]] {
//         auto &d_pre = *prev;
//         auto &perm = utils::permutation_from_y_to_x(prev->dense_->prob_, cur->dense_->prob_);
//         d.Q_x *= perm;
//         d_pre.Q_y += d.Q_x;
//     }
// }
// } // namespace lcid_riccati
// } // namespace solver
// } // namespace moto