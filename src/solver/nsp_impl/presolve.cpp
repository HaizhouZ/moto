#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

// #define SHOW_NSP_DEBUG

namespace moto {
namespace solver {
namespace ns_riccati {
extern void print_debug(ns_riccati_data *cur);

void generic_solver::ns_factorization_correction(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    auto &_approx = d.dense_->approx_;

    cur->update_projected_dynamics_residual();

    nsp.u_0_p_k = d.Q_u.transpose();
    nsp.y_0_p_k.setZero();
    nsp.y_y_k = d.F_0;

    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_0_k = nsp.u_0_p_k;
        d.Q_yx.right_T_times<false>(d.F_0, d.Q_x);
        d.Q_yx_mod.right_T_times<false>(d.F_0, d.Q_x);
        return;
    }

    auto constr_s = d.ns + d.nis;
    auto constr_c = d.nc + d.nic;
    d.ncstr = constr_s + constr_c;

    nsp.s_c_stacked_0_k.conservativeResize(d.ncstr);
    timed_block_start("setup ns residuals");
    if (constr_s) {
        nsp.s_0_p_k.conservativeResize(constr_s);
        nsp.s_0_p_k.noalias() = _approx[__eq_x].v_;
        d.s_y.times<false>(d.F_0, nsp.s_0_p_k);
        nsp.s_c_stacked_0_k.head(constr_s) = nsp.s_0_p_k;
    }
    if (constr_c) {
        nsp.s_c_stacked_0_k.tail(d.nc) = _approx[__eq_xu].v_;
    }
    timed_block_end("setup ns residuals");

    timed_block_start("precompute_u_y");
    nsp.u_y_k.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_k);
    timed_block_end("precompute_u_y");

    timed_block_start("precompute_u0p");
    d.Q_uu.times<false>(nsp.u_y_k, nsp.u_0_p_k);
    d.Q_uu_mod.times<false>(nsp.u_y_k, nsp.u_0_p_k);
    timed_block_end("precompute_u0p");

    timed_block_start("precompute_z0");
    if (d.rank_status_ != rank_status::fully_constrained) {
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_k;
    }
    timed_block_end("precompute_z0");

    timed_block_start("precompute_y_y");
    d.F_u.times<false>(nsp.u_y_k, nsp.y_y_k);
    timed_block_end("precompute_y_y");

    d.Q_x.noalias() -= nsp.u_0_p_k.transpose() * nsp.u_y_K;
    d.Q_ux.right_T_times<false>(nsp.u_y_k, d.Q_x);
    d.Q_ux_mod.right_T_times<false>(nsp.u_y_k, d.Q_x);
    d.Q_yx.right_T_times<false>(nsp.y_y_k, d.Q_x);
    d.Q_yx_mod.right_T_times<false>(nsp.y_y_k, d.Q_x);

    if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.k = -nsp.u_y_k;
        d.d_y.k = -nsp.y_y_k;
    }
}

void generic_solver::ns_factorization(ns_riccati_data *cur, bool gauss_newton) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    auto &_approx = d.dense_->approx_;

    timed_block_start("update_projected_dynamics");
    cur->update_projected_dynamics();
    timed_block_end("update_projected_dynamics");

    nsp.u_0_p_K.setZero();
    d.Q_ux.dump_into(nsp.u_0_p_K);
    d.Q_ux_mod.dump_into(nsp.u_0_p_K);
    nsp.y_0_p_K.setZero();
    d.Q_yx.dump_into(nsp.y_0_p_K);
    d.Q_yx_mod.dump_into(nsp.y_0_p_K);
    d.V_xx.setZero();
    d.Q_xx.dump_into(d.V_xx);
    d.Q_xx_mod.dump_into(d.V_xx);
    d.V_yy.setZero();
    d.Q_yy.dump_into(d.V_yy);
    d.Q_yy_mod.dump_into(d.V_yy);

    d.rank_status_ = rank_status::unconstrained;

    auto unconstrain_setup = [&]() {
        d.rank_status_ = rank_status::unconstrained;
        nsp.z_k.conservativeResize(d.nu);
        nsp.z_K.conservativeResize(d.nu, d.nx);
        nsp.z_0_k.conservativeResize(d.nu);
        nsp.z_0_K.conservativeResize(d.nu, d.nx);
        nsp.Q_zz.conservativeResize(d.nu, d.nu);
        nsp.Q_zz.setZero();
        d.Q_uu.dump_into(nsp.Q_zz);
        d.Q_uu_mod.dump_into(nsp.Q_zz);
        nsp.z_0_K = nsp.u_0_p_K;
        d.F_x.T_times<false>(d.Q_yx, d.V_xx);
        d.F_x.T_times<false>(d.Q_yx_mod, d.V_xx);
    };

    auto activate_gradient_corrections = [&]() {
        timed_block_start("activate_lag_jac_corr");
        cur->activate_lag_jac_corr();
        timed_block_end("activate_lag_jac_corr");
    };

    if (!d.ncstr) {
        unconstrain_setup();
        activate_gradient_corrections();
        ns_factorization_correction(cur);
        return;
    }

    d.nis = 0;
    d.nic = 0;
    size_t constr_s = d.ns + d.nis;
    size_t constr_c = d.nc + d.nic;
    d.ncstr = constr_s + constr_c;
    nsp.s_c_stacked.conservativeResize(d.ncstr, Eigen::NoChange);
    nsp.s_c_stacked.setZero();
    d.d_lbd_s_c.conservativeResize(d.ncstr);

    timed_block_start("copy_lhs_derivatives");
    if (constr_s) {
        nsp.s_u.setZero();
        d.s_y.times<false>(d.F_u, nsp.s_u);
        nsp.s_c_stacked.topRows(constr_s) = nsp.s_u;
    }
    if (constr_c) {
        d.c_u.dump_into(nsp.s_c_stacked.bottomRows(d.nc), spmm::dump_config{.overwrite = true});
    }
    timed_block_end("copy_lhs_derivatives");

    timed_block_start("lu_eq_compute");
    nsp.lu_eq_.compute(nsp.s_c_stacked);
    nsp.rank = nsp.lu_eq_.rank();
    timed_block_end("lu_eq_compute");

    // build s_c_stacked_0_K (needed for GN mode and for the normal constrained path)
    nsp.s_c_stacked_0_K.conservativeResize(d.ncstr, Eigen::NoChange);
    nsp.s_c_stacked_0_K.setZero();
    if (constr_s) {
        nsp.s_0_p_K.conservativeResize(constr_s, Eigen::NoChange);
        nsp.s_0_p_K.setZero();
        d.s_x.dump_into(nsp.s_0_p_K);
        d.s_y.times<false>(d.F_x, nsp.s_0_p_K);
        nsp.s_c_stacked_0_K.topRows(constr_s) = nsp.s_0_p_K;
    }
    if (constr_c) {
        d.c_x.dump_into(nsp.s_c_stacked_0_K.bottomRows(d.nc));
    }

    auto &rank = nsp.rank;
    if (rank == 0) {
        d.rank_status_ = rank_status::unconstrained;
        unconstrain_setup();
        activate_gradient_corrections();
        ns_factorization_correction(cur);
        return;
    }

    if (rank == d.nu) {
        d.rank_status_ = rank_status::fully_constrained;
    } else {
        timed_block_start("compute_nullspace");
        nsp.Z_u = nsp.lu_eq_.kernel();
        timed_block_end("compute_nullspace");

        timed_block_start("compute_Zy");
        nsp.Z_y.resize(d.ny, nsp.Z_u.cols());
        nsp.Z_y.setZero();
        d.F_u.times<false>(nsp.Z_u, nsp.Z_y);
        timed_block_end("compute_Zy");

        d.rank_status_ = rank_status::constrained;
        timed_block_start("compute_Qzz");
        nsp.Q_zz.conservativeResize(nsp.Z_u.cols(), nsp.Z_u.cols());
        thread_local moto::utils::buffer_tpl<matrix> buf;
        buf.resize(d.nu, nsp.Z_u.cols());
        buf.data_.setZero();
        d.Q_uu.times(nsp.Z_u, buf.data_);
        d.Q_uu_mod.times(nsp.Z_u, buf.data_);
        nsp.Q_zz.noalias() = nsp.Z_u.transpose() * buf.data_;
        timed_block_end("compute_Qzz");

        nsp.z_k.conservativeResize(nsp.Z_u.cols());
        nsp.z_K.conservativeResize(nsp.Z_u.cols(), d.nx);
        nsp.z_0_k.conservativeResize(nsp.Z_u.cols());
        nsp.z_0_K.conservativeResize(nsp.Z_u.cols(), d.nx);
    }

    if (d.rank_status_ != rank_status::unconstrained) {
        timed_block_start("precompute_u_y");
        nsp.u_y_K.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_K);
        timed_block_end("precompute_u_y");

        timed_block_start("precompute_u0p");
        d.Q_uu.times<false>(nsp.u_y_K, nsp.u_0_p_K);
        d.Q_uu_mod.times<false>(nsp.u_y_K, nsp.u_0_p_K);
        timed_block_end("precompute_u0p");

        timed_block_start("precompute_z0");
        if (d.rank_status_ != rank_status::fully_constrained) {
            nsp.z_0_K.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_K;
        }
        timed_block_end("precompute_z0");

        timed_block_start("precompute_y_y");
        nsp.y_y_K.setZero();
        d.F_x.dump_into(nsp.y_y_K);
        d.F_u.times<false>(nsp.u_y_K, nsp.y_y_K);
        timed_block_end("precompute_y_y");

        timed_block_start("update_value_derivative");
        d.V_xx.noalias() -= nsp.u_0_p_K.transpose() * nsp.u_y_K;
        d.Q_ux.right_T_times<false>(nsp.u_y_K, d.V_xx);
        d.Q_ux_mod.right_T_times<false>(nsp.u_y_K, d.V_xx);
        d.Q_yx.right_T_times<false>(nsp.y_y_K, d.V_xx);
        d.Q_yx_mod.right_T_times<false>(nsp.y_y_K, d.V_xx);
        timed_block_end("update_value_derivative");
    }

    if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.K = -nsp.u_y_K;
        d.d_y.K = -nsp.y_y_K;
    }
    activate_gradient_corrections();
    ns_factorization_correction(cur);
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
