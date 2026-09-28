#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/core/linear_backend.hpp>

#include <cstdlib>
#include <stdexcept>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void generic_solver::ns_factorization_correction(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;

    if (!(d.has_integrated_presolve_graph() && !d.ncstr))
        d.update_projected_dynamics_residual();

    nsp.u_0_p_k = d.Q_u.transpose();
    nsp.y_0_p_k.setZero();
    nsp.l_0_p_k = d.Q_l.transpose();

    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.u_y_k.setZero();
    } else {
        nsp.s_c_stacked_0_k.conservativeResize(d.ncstr);
        d.build_lifted_hard_geometry(nullptr, nullptr, &nsp.s_c_stacked_0_k);
        nsp.u_y_k.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_k);
    }
    d.update_lifted_basis_k();

    nsp.z_0_k.setZero();
    if (d.rank_status_ == rank_status::unconstrained)
        nsp.z_0_k = d.Q_u.transpose();
    else
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * d.Q_u.transpose();
    if (d.uses_sparse_lifted_basis())
        linear_backend::transpose_multiply(
            d.lifting_.l_u(), nsp.l_0_p_k, nsp.z_0_k, -1.);
    else
        nsp.z_0_k.noalias() += nsp.Z_l.transpose() * d.Q_l.transpose();
    d.Q_x.noalias() -= d.Q_u * nsp.u_y_K;
    if (d.uses_sparse_lifted_basis())
        linear_backend::right_multiply(
            d.Q_l, d.lifting_.l_x(), d.Q_x, -1.);
    else
        d.Q_x.noalias() -= d.Q_l * nsp.l_y_K;

    if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.k = -nsp.u_y_k;
        d.d_y.k = -nsp.y_y_k;
    }
}

void generic_solver::ns_factorization(ns_riccati_data *cur, bool gauss_newton) {
    auto &d = *cur;
    auto &nsp = d.nsp_;

    const bool integrated_presolve =
        d.has_integrated_presolve_graph() && !d.ns && !d.nc;
    const bool verify_graph =
        integrated_presolve && std::getenv("MOTO_VERIFY_NSP_GRAPH");
    timed_block_start("update_projected_dynamics");
    if (!integrated_presolve || verify_graph)
        d.update_projected_dynamics();
    timed_block_end("update_projected_dynamics");

    d.V_yy.setZero();
    linear_backend::write_dense(d.Q_yy, d.V_yy);
    linear_backend::write_dense(d.Q_yy_mod, d.V_yy);

    d.nis = 0;
    d.nic = 0;
    d.ncstr = d.ns + d.nc;
    d.rank_status_ = rank_status::unconstrained;
    nsp.rank = 0;

    if (d.ncstr) {
        nsp.s_c_stacked.conservativeResize(d.ncstr, d.nu);
        d.build_lifted_hard_geometry(&nsp.s_c_stacked, nullptr, nullptr);
        nsp.lu_eq_.compute(nsp.s_c_stacked);
        nsp.rank = nsp.lu_eq_.rank();
    }

    if (!nsp.rank) {
        nsp.Z_u.resize(d.nu, 0);
    } else if (nsp.rank == d.nu) {
        d.rank_status_ = rank_status::fully_constrained;
        nsp.Z_u.resize(d.nu, 0);
    } else {
        d.rank_status_ = rank_status::constrained;
        nsp.Z_u = nsp.lu_eq_.kernel();
    }

    const size_t nz =
        d.rank_status_ == rank_status::unconstrained ? d.nu : nsp.Z_u.cols();
    nsp.Z_y.resize(d.ny, nz);
    nsp.Z_l.resize(d.nl, nz);
    nsp.z_k.conservativeResize(nz);
    nsp.z_K.conservativeResize(nz, d.nx);
    nsp.z_0_k.conservativeResize(nz);
    nsp.z_0_K.conservativeResize(nz, d.nx);
    nsp.Q_zz.setZero(nz, nz);

    nsp.s_c_stacked_0_K.conservativeResize(d.ncstr, d.nx);
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.u_y_K.setZero();
    } else {
        d.build_lifted_hard_geometry(nullptr, &nsp.s_c_stacked_0_K, nullptr);
        nsp.u_y_K.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_K);
    }
    if (!integrated_presolve || verify_graph)
        d.update_lifted_basis_K();

    const bool graph_presolve = d.rank_status_ == rank_status::unconstrained &&
                                !d.ncstr &&
                                d.has_unconstrained_presolve_graph();
    matrix expected_qzz, expected_z_0_K, expected_v_xx;
    if (!graph_presolve || verify_graph) {
        d.V_xx.setZero();
        linear_backend::write_dense(d.Q_xx, d.V_xx);
        linear_backend::write_dense(d.Q_xx_mod, d.V_xx);
        if (d.rank_status_ == rank_status::unconstrained) {
            nsp.u_0_p_K.setZero(d.nu, nz);
            linear_backend::write_dense(d.Q_uu, nsp.u_0_p_K);
            linear_backend::write_dense(d.Q_uu_mod, nsp.u_0_p_K);
        } else {
            nsp.u_0_p_K.setZero(d.nu, nz);
            linear_backend::multiply(d.Q_uu, nsp.Z_u, nsp.u_0_p_K);
            linear_backend::multiply(d.Q_uu_mod, nsp.Z_u, nsp.u_0_p_K);
        }
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__y][__u],
                                           nsp.Z_y, nsp.u_0_p_K);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__y][__u], nsp.Z_y, nsp.u_0_p_K);
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__u],
                                           nsp.Z_l, nsp.u_0_p_K);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__l][__u], nsp.Z_l, nsp.u_0_p_K);

        nsp.y_0_p_K.setZero(d.ny, nz);
        if (d.rank_status_ == rank_status::unconstrained) {
            linear_backend::write_dense(d.dense_->lag_hess_[__y][__u],
                                        nsp.y_0_p_K);
            linear_backend::write_dense(
                d.dense_->hessian_modification_[__y][__u], nsp.y_0_p_K);
        } else {
            linear_backend::multiply(d.dense_->lag_hess_[__y][__u], nsp.Z_u,
                                     nsp.y_0_p_K);
            linear_backend::multiply(d.dense_->hessian_modification_[__y][__u],
                                     nsp.Z_u, nsp.y_0_p_K);
        }
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__y],
                                           nsp.Z_l, nsp.y_0_p_K);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__l][__y], nsp.Z_l, nsp.y_0_p_K);

        nsp.l_0_p_K.setZero(d.nl, nz);
        if (d.rank_status_ == rank_status::unconstrained) {
            linear_backend::write_dense(d.dense_->lag_hess_[__l][__u],
                                        nsp.l_0_p_K);
            linear_backend::write_dense(
                d.dense_->hessian_modification_[__l][__u], nsp.l_0_p_K);
        } else {
            linear_backend::multiply(d.dense_->lag_hess_[__l][__u], nsp.Z_u,
                                     nsp.l_0_p_K);
            linear_backend::multiply(d.dense_->hessian_modification_[__l][__u],
                                     nsp.Z_u, nsp.l_0_p_K);
        }
        linear_backend::multiply(d.dense_->lag_hess_[__l][__y], nsp.Z_y,
                                 nsp.l_0_p_K);
        linear_backend::multiply(d.dense_->hessian_modification_[__l][__y],
                                 nsp.Z_y, nsp.l_0_p_K);
        linear_backend::multiply(d.Q_ll, nsp.Z_l, nsp.l_0_p_K);
        linear_backend::multiply(d.Q_ll_mod, nsp.Z_l, nsp.l_0_p_K);

        if (d.rank_status_ == rank_status::unconstrained)
            nsp.Q_zz = nsp.u_0_p_K;
        else
            nsp.Q_zz.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_K;
        nsp.Q_zz.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_K;
        nsp.Q_zz.noalias() += nsp.Z_l.transpose() * nsp.l_0_p_K;
        nsp.u_0_p_K.setZero(d.nu, d.nx);
        linear_backend::write_dense(d.Q_ux, nsp.u_0_p_K);
        linear_backend::write_dense(d.Q_ux_mod, nsp.u_0_p_K);
        linear_backend::multiply(d.Q_uu, nsp.u_y_K, nsp.u_0_p_K, -1.);
        linear_backend::multiply(d.Q_uu_mod, nsp.u_y_K, nsp.u_0_p_K, -1.);
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__y][__u],
                                           nsp.y_y_K, nsp.u_0_p_K, -1.);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__y][__u], nsp.y_y_K, nsp.u_0_p_K,
            -1.);
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__u],
                                           nsp.l_y_K, nsp.u_0_p_K, -1.);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__l][__u], nsp.l_y_K, nsp.u_0_p_K,
            -1.);

        nsp.y_0_p_K.setZero(d.ny, d.nx);
        linear_backend::write_dense(d.Q_yx, nsp.y_0_p_K);
        linear_backend::write_dense(d.Q_yx_mod, nsp.y_0_p_K);
        linear_backend::multiply(d.dense_->lag_hess_[__y][__u], nsp.u_y_K,
                                 nsp.y_0_p_K, -1.);
        linear_backend::multiply(d.dense_->hessian_modification_[__y][__u],
                                 nsp.u_y_K, nsp.y_0_p_K, -1.);
        linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__y],
                                           nsp.l_y_K, nsp.y_0_p_K, -1.);
        linear_backend::transpose_multiply(
            d.dense_->hessian_modification_[__l][__y], nsp.l_y_K, nsp.y_0_p_K,
            -1.);

        nsp.l_0_p_K.setZero(d.nl, d.nx);
        linear_backend::write_dense(d.Q_lx, nsp.l_0_p_K);
        linear_backend::write_dense(d.Q_lx_mod, nsp.l_0_p_K);
        linear_backend::multiply(d.dense_->lag_hess_[__l][__u], nsp.u_y_K,
                                 nsp.l_0_p_K, -1.);
        linear_backend::multiply(d.dense_->hessian_modification_[__l][__u],
                                 nsp.u_y_K, nsp.l_0_p_K, -1.);
        linear_backend::multiply(d.dense_->lag_hess_[__l][__y], nsp.y_y_K,
                                 nsp.l_0_p_K, -1.);
        linear_backend::multiply(d.dense_->hessian_modification_[__l][__y],
                                 nsp.y_y_K, nsp.l_0_p_K, -1.);
        linear_backend::multiply(d.Q_ll, nsp.l_y_K, nsp.l_0_p_K, -1.);
        linear_backend::multiply(d.Q_ll_mod, nsp.l_y_K, nsp.l_0_p_K, -1.);

        if (d.rank_status_ == rank_status::unconstrained)
            nsp.z_0_K = nsp.u_0_p_K;
        else
            nsp.z_0_K.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_K;
        nsp.z_0_K.noalias() += nsp.Z_y.transpose() * nsp.y_0_p_K;
        nsp.z_0_K.noalias() += nsp.Z_l.transpose() * nsp.l_0_p_K;

        linear_backend::transpose_multiply(d.Q_ux, nsp.u_y_K, d.V_xx, -1.);
        linear_backend::transpose_multiply(d.Q_ux_mod, nsp.u_y_K, d.V_xx, -1.);
        linear_backend::transpose_multiply(d.Q_yx, nsp.y_y_K, d.V_xx, -1.);
        linear_backend::transpose_multiply(d.Q_yx_mod, nsp.y_y_K, d.V_xx, -1.);
        linear_backend::transpose_multiply(d.Q_lx, nsp.l_y_K, d.V_xx, -1.);
        linear_backend::transpose_multiply(d.Q_lx_mod, nsp.l_y_K, d.V_xx, -1.);
        d.V_xx.noalias() -= nsp.u_y_K.transpose() * nsp.u_0_p_K;
        d.V_xx.noalias() -= nsp.y_y_K.transpose() * nsp.y_0_p_K;
        d.V_xx.noalias() -= nsp.l_y_K.transpose() * nsp.l_0_p_K;
        if (verify_graph) {
            expected_qzz = nsp.Q_zz;
            expected_z_0_K = nsp.z_0_K;
            expected_v_xx = d.V_xx;
        }
    }
    if (graph_presolve) {
        d.run_unconstrained_presolve_graph();
        if (verify_graph && (!nsp.Q_zz.isApprox(expected_qzz, 1e-10) ||
                             !nsp.z_0_K.isApprox(expected_z_0_K, 1e-10) ||
                             !d.V_xx.isApprox(expected_v_xx, 1e-10)))
            throw std::runtime_error(
                fmt::format("NSP presolve graph mismatch: Qzz={} z0K={} Vxx={}",
                            (nsp.Q_zz - expected_qzz).cwiseAbs().maxCoeff(),
                            (nsp.z_0_K - expected_z_0_K).cwiseAbs().maxCoeff(),
                            (d.V_xx - expected_v_xx).cwiseAbs().maxCoeff()));
    }

    d.activate_lag_jac_corr();
    ns_factorization_correction(cur);

    d.Q_x.noalias() += d.Q_u * nsp.u_y_K;
    if (d.uses_sparse_lifted_basis())
        linear_backend::right_multiply(d.Q_l, d.lifting_.l_x(), d.Q_x);
    else
        d.Q_x.noalias() += d.Q_l * nsp.l_y_K;
    nsp.u_0_p_k = d.Q_u.transpose();
    linear_backend::multiply(d.Q_uu, nsp.u_y_k, nsp.u_0_p_k, -1.);
    linear_backend::multiply(d.Q_uu_mod, nsp.u_y_k, nsp.u_0_p_k, -1.);
    linear_backend::transpose_multiply(d.dense_->lag_hess_[__y][__u],
                                       nsp.y_y_k, nsp.u_0_p_k, -1.);
    linear_backend::transpose_multiply(
        d.dense_->hessian_modification_[__y][__u], nsp.y_y_k,
        nsp.u_0_p_k, -1.);
    linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__u],
                                       nsp.l_y_k, nsp.u_0_p_k, -1.);
    linear_backend::transpose_multiply(
        d.dense_->hessian_modification_[__l][__u], nsp.l_y_k,
        nsp.u_0_p_k, -1.);

    nsp.y_0_p_k.setZero();
    linear_backend::multiply(d.dense_->lag_hess_[__y][__u], nsp.u_y_k,
                             nsp.y_0_p_k, -1.);
    linear_backend::multiply(d.dense_->hessian_modification_[__y][__u],
                             nsp.u_y_k, nsp.y_0_p_k, -1.);
    linear_backend::transpose_multiply(d.dense_->lag_hess_[__l][__y], nsp.l_y_k,
                                       nsp.y_0_p_k, -1.);
    linear_backend::transpose_multiply(
        d.dense_->hessian_modification_[__l][__y], nsp.l_y_k, nsp.y_0_p_k, -1.);

    nsp.l_0_p_k = d.Q_l.transpose();
    linear_backend::multiply(d.dense_->lag_hess_[__l][__u], nsp.u_y_k,
                             nsp.l_0_p_k, -1.);
    linear_backend::multiply(d.dense_->hessian_modification_[__l][__u],
                             nsp.u_y_k, nsp.l_0_p_k, -1.);
    linear_backend::multiply(d.dense_->lag_hess_[__l][__y], nsp.y_y_k,
                             nsp.l_0_p_k, -1.);
    linear_backend::multiply(d.dense_->hessian_modification_[__l][__y],
                             nsp.y_y_k, nsp.l_0_p_k, -1.);
    linear_backend::multiply(d.Q_ll, nsp.l_y_k, nsp.l_0_p_k, -1.);
    linear_backend::multiply(d.Q_ll_mod, nsp.l_y_k, nsp.l_0_p_k, -1.);

    if (d.rank_status_ == rank_status::unconstrained)
        nsp.z_0_k = nsp.u_0_p_k;
    else
        nsp.z_0_k.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_k;
    if (d.uses_sparse_lifted_basis())
        linear_backend::transpose_multiply(
            d.lifting_.l_u(), nsp.l_0_p_k, nsp.z_0_k, -1.);
    else
        nsp.z_0_k.noalias() += nsp.Z_l.transpose() * nsp.l_0_p_k;
    d.Q_x.noalias() -= nsp.u_0_p_k.transpose() * nsp.u_y_K;
    if (d.uses_sparse_lifted_basis())
        linear_backend::right_multiply(
            nsp.l_0_p_k.transpose(), d.lifting_.l_x(), d.Q_x, -1.);
    else
        d.Q_x.noalias() -= nsp.l_0_p_k.transpose() * nsp.l_y_K;
    linear_backend::transpose_multiply(d.Q_ux, nsp.u_y_k, d.Q_x, -1.);
    linear_backend::transpose_multiply(d.Q_ux_mod, nsp.u_y_k, d.Q_x, -1.);
    linear_backend::transpose_multiply(d.Q_yx, nsp.y_y_k, d.Q_x, -1.);
    linear_backend::transpose_multiply(d.Q_yx_mod, nsp.y_y_k, d.Q_x, -1.);
    linear_backend::transpose_multiply(d.Q_lx, nsp.l_y_k, d.Q_x, -1.);
    linear_backend::transpose_multiply(d.Q_lx_mod, nsp.l_y_k, d.Q_x, -1.);
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto
