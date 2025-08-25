#include <Eigen/Eigenvalues>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>
// #define SHOW_NSP_DEBUG

namespace moto {
namespace solver {
namespace ns_riccati {
extern void print_debug(ns_node_data *cur);

void ns_factorization(ns_node_data *cur) {
    // collect constraint residuals and jacobians
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    auto &_approx = d.dense_->approx_;
    cur->update_projected_dynamics();
    // auto &F = _approx[__dyn].jac_;
    // partial value derivative
    cur->merge_jacobian_modification();
    // d.Q_x.noalias() += -d.F_0.transpose() * d.Q_yx;
    // nsp.Q_yy_F_x = -d.Q_yx;
    nsp.u_0_p_k = d.Q_u.transpose();
    nsp.u_0_p_K = d.Q_ux;
    nsp.y_0_p_k.setZero();
    nsp.y_0_p_K = d.Q_yx;
    // check Q_xx symmetry
    // if ((d.Q_xx - d.Q_xx.transpose()).cwiseAbs().maxCoeff() > 1e-10) {
    //     fmt::print("Q_xx is not symmetric before nsp factorization: max abs diff = {}\n", (d.Q_xx - d.Q_xx.transpose()).cwiseAbs().maxCoeff());
    //     throw std::runtime_error("Q_xx is not symmetric");
    // }
    // if (d.F_x.dense().array().isNaN().any() || d.Q_xx.array().isNaN().any()) {
    //     fmt::print("F_0_K:{}\n", d.dense_->approx_[__dyn].jac_[__x].dense());
    //     fmt::print("F_0_K:{}\n", d.dense_->approx_[__dyn].jac_[__y].dense());
    //     fmt::print("Q_xx:{}\n", d.Q_xx);
    //     throw std::runtime_error("NaN detected in F_0_K or Q_xx");
    // }
    // d.Q_xx.noalias() += -(d.Q_yx.transpose() * d.F_x.dense() + d.F_x.dense().transpose() * d.Q_yx);
    // d.F_x.right_T_times<false>(d.Q_yx, d.Q_xx);
    // nullspace computation
    d.rank_status_ = rank_status::unconstrained;

    auto unconstrain_setup = [&]() {
        d.rank_status_ = rank_status::unconstrained;
        nsp.z_k.conservativeResize(d.nu);
        nsp.z_K.conservativeResize(d.nu, d.nx);
        nsp.z_0_k.conservativeResize(d.nu);
        nsp.z_0_K.conservativeResize(d.nu, d.nx);
        nsp.Q_zz.conservativeResize(d.nu, d.nu);
        nsp.Q_zz = d.Q_uu;
        nsp.z_0_k = nsp.u_0_p_k;
        nsp.z_0_K = nsp.u_0_p_K;
        d.Q_x.noalias() -= d.F_0.transpose() * d.Q_yx;
        d.F_x.T_times<false>(d.Q_yx, d.Q_xx);
    };
    if (!d.ncstr) {
        unconstrain_setup();
    } else {
        d.nis = 0;
        d.nic = 0;
        size_t constr_s = d.ns + d.nis;
        size_t constr_c = d.nc + d.nic;
        d.ncstr = constr_s + constr_c;
        nsp.s_c_stacked.conservativeResize(d.ncstr, Eigen::NoChange);
        nsp.s_c_stacked.setZero();
        d.d_lbd_s_c.conservativeResize(d.ncstr);

        if (constr_s) {
            nsp.s_u.setZero();
            d.s_y.times<false>(d.F_u, nsp.s_u);
            // nsp.s_u.noalias() = -nsp.s_y * nsp.F_u;
            // solve pseudo inverse
            nsp.s_c_stacked.topRows(constr_s) = nsp.s_u;
        }
        if (constr_c) {
            d.c_u.dump_into(nsp.s_c_stacked.bottomRows(d.nc), spmm::dump_config{.overwrite = true});
            // nsp.s_c_stacked.bottomRows(d.nc) = _approx[__eq_xu].jac_[__u];
        }
        nsp.lu_eq_.compute(nsp.s_c_stacked);
        nsp.rank = nsp.lu_eq_.rank();
        auto &rank = nsp.rank;
#ifdef SHOW_NSP_DEBUG
        fmt::print("rank of equality constraints: {}\n", rank);
#endif

        if (rank == 0) {
            d.rank_status_ = rank_status::unconstrained;
            unconstrain_setup();
        } else {
            if (rank == d.nu) {
                d.rank_status_ = rank_status::fully_constrained;
#ifdef SHOW_NSP_DEBUG
                fmt::print("warning: fully constrained node detected\n");
#endif
            } else {
                nsp.Z_u = nsp.lu_eq_.kernel();
                // auto &cod = nsp.lu_eq_;
                // matrix V = cod.matrixZ().transpose();
                // matrix Null_space = V.block(0, cod.rank(), V.rows(), V.cols() - cod.rank());
                // matrix P = cod.colsPermutation();
                // size_t nullity = Null_space.cols();
                // nsp.Z_u.resize(nsp.s_c_stacked.cols(), Null_space.cols());
                // nsp.Z_u = P * Null_space;

                // if (!(nsp.Z_u.transpose() * nsp.Z_u).isIdentity(1e-6)) {
                //     fmt::print("scstacked_ \n{}\n", nsp.s_c_stacked);
                //     fmt::print("Z_u:\n{}\n", ((nsp.Z_u.transpose() * nsp.Z_u) - Eigen::MatrixXd::Identity(nullity, nullity)).cwiseAbs());
                //     throw std::runtime_error("Numerical issue detected: nullspace basis is not orthonormal");
                // }
                nsp.Z_y.resize(d.ny, nsp.Z_u.cols());
                nsp.Z_y.setZero();
                d.F_u.times<false>(nsp.Z_u, nsp.Z_y);

                // matrix stacked_Z = matrix::Zero(d.nu + d.ny, nsp.Z_u.cols());
                // stacked_Z << nsp.Z_u, nsp.Z_y;
                // if (!(nsp.Z_y.transpose() * nsp.Z_y).isIdentity(1e-6)) {
                // // if (!(stacked_Z.transpose() * stacked_Z).isIdentity(1e-6)) {
                //     fmt::print("scstacked_ \n{}\n", nsp.s_c_stacked);
                //     fmt::print("Z_y:\n{}\n", ((nsp.Z_y.transpose() * nsp.Z_y) - Eigen::MatrixXd::Identity(nullity, nullity)).cwiseAbs());
                //     // fmt::print("Z_u:\n{}\n", ((stacked_Z.transpose() * stacked_Z) - Eigen::MatrixXd::Identity(nullity, nullity)).cwiseAbs());
                //     throw std::runtime_error("Numerical issue detected: nullspace basis is not orthonormal");
                // }

#ifdef SHOW_NSP_DEBUG
                // check nullspace kernel
                matrix stacked_ = matrix::Zero(d.ny + d.ncstr, d.nu + d.ny);
                stacked_.topRows(d.ny) << cur->dense_->approx_[__dyn].jac_[__u].dense(), cur->dense_->approx_[__dyn].jac_[__y].dense();
                stacked_.middleRows(d.ny, d.ns).rightCols(d.ny) = cur->dense_->approx_[__eq_x].jac_[__y].dense();
                if (d.nc)
                    stacked_.bottomRows(d.nc).leftCols(d.nu) = cur->dense_->approx_[__eq_xu].jac_[__u].dense();
                // fmt::print("Zy {}\n", nsp.Z_y.cwiseAbs().maxCoeff());
                // fmt::print("Zu {}\n", nsp.Z_u.cwiseAbs().maxCoeff());
                // fmt::print("stacked_ \n{}\n", cur->dense_->approx_[__eq_x].jac_[__y].dense());
                if (stacked_.cwiseAbs().maxCoeff() > 1e10) {
                    throw std::runtime_error("Numerical issue detected: very large constraint Jacobian");
                }
                // fmt::print("stacked_Z {}\n", stacked_Z.cwiseAbs().maxCoeff());
                fmt::print("kernel residual u = {}\n", (nsp.s_c_stacked * nsp.Z_u).cwiseAbs().maxCoeff());
                // fmt::print("kernel residual = {}\n", (stacked_ * stacked_Z).cwiseAbs().maxCoeff());

#endif
                // fmt::print("nullspace :\n {}\n", nsp.Z.cols());
                d.rank_status_ = rank_status::constrained;
                nsp.Q_zz.conservativeResize(nsp.Z_u.cols(), nsp.Z_u.cols());
                nsp.Q_zz.noalias() = nsp.Z_u.transpose() * d.Q_uu * nsp.Z_u;
                nsp.z_k.conservativeResize(nsp.Z_u.cols());
                nsp.z_K.conservativeResize(nsp.Z_u.cols(), d.nx);
                nsp.z_0_k.conservativeResize(nsp.Z_u.cols());
                nsp.z_0_K.conservativeResize(nsp.Z_u.cols(), d.nx);
            }
        }
        // precompute
        nsp.s_c_stacked_0_k.conservativeResize(d.ncstr);
        nsp.s_c_stacked_0_K.conservativeResize(d.ncstr, Eigen::NoChange);
        nsp.s_c_stacked_0_K.setZero();
        if (constr_s) {
            // nsp.s_0_p_k.noalias() =
            //     _approx[__eq_x].v_ - nsp.s_y * nsp.F_0_k;
            nsp.s_0_p_k.conservativeResize(constr_s);
            nsp.s_0_p_k.noalias() = _approx[__eq_x].v_;
            d.s_y.times<false>(d.F_0, nsp.s_0_p_k);
            // nsp.s_0_p_K.noalias() =
            //     _approx[__eq_x].jac_[__x] - nsp.s_y * nsp.F_0_K;
            nsp.s_0_p_K.conservativeResize(constr_s, Eigen::NoChange);
            nsp.s_0_p_K.setZero();
            d.s_x.dump_into(nsp.s_0_p_K);
            d.s_y.times<false>(d.F_x, nsp.s_0_p_K);

            nsp.s_c_stacked_0_k.head(constr_s) = nsp.s_0_p_k;
            nsp.s_c_stacked_0_K.topRows(constr_s) = nsp.s_0_p_K;
        }
        if (constr_c) {
            nsp.s_c_stacked_0_k.tail(d.nc) = _approx[__eq_xu].v_;
            // nsp.s_c_stacked_0_K.bottomRows(d.nc) = _approx[__eq_xu].jac_[__x];
            d.c_x.dump_into(nsp.s_c_stacked_0_K.bottomRows(d.nc));
        }
        if (d.rank_status_ != rank_status::unconstrained) {
            // pre compute
            nsp.u_y_k.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_k);
            nsp.u_y_K.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_K);
            nsp.u_0_p_k.noalias() -= d.Q_uu * nsp.u_y_k;
            nsp.u_0_p_K.noalias() -= d.Q_uu * nsp.u_y_K;
            if (d.rank_status_ != rank_status::fully_constrained) {
                nsp.z_0_k.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_k;
                nsp.z_0_K.noalias() = nsp.Z_u.transpose() * nsp.u_0_p_K;
            }
            // fmt::print("jac: \n {}\n", nsp.s_c_stacked);
            // fmt::print("nyk :\n {}\n", nsp.u_y_k.transpose());
            // fmt::print("scstacked_0_K :\n {}\n", nsp.s_c_stacked_0_K);
            // fmt::print("nyK :\n {}\n", nsp.u_y_K);
            // fmt::print("F_u: \n{}\n", nsp.F_u.transpose());
            // fmt::print("F_0_K: \n{}\n", nsp.F_0_K.transpose());
            nsp.y_y_k = d.F_0;
            d.F_u.times<false>(nsp.u_y_k, nsp.y_y_k);
            nsp.y_y_K.setZero();
            d.F_x.dump_into(nsp.y_y_K);
            d.F_u.times<false>(nsp.u_y_K, nsp.y_y_K);

// print_debug(cur);
#ifdef SHOW_NSP_DEBUG
            fmt::print("scstacked :\n {}\n", nsp.s_c_stacked);
            fmt::print("scstacked_0_k :\n {}\n", nsp.s_c_stacked_0_k.transpose());
            // Compute dynamics residual using dense_->approx
            fmt::print("u_y_k: \n{}\n", nsp.u_y_k.transpose());
            fmt::print("u_y_K: \n{}\n", nsp.u_y_K);
            fmt::print("lu residual = {}\n", (nsp.s_c_stacked * nsp.u_y_k - nsp.s_c_stacked_0_k).cwiseAbs().maxCoeff());
            vector proj_dyn_res = d.F_0 - d.F_u.dense() * nsp.u_y_k - nsp.y_y_k;
            fmt::print("projected dynamics residual: {}\n", proj_dyn_res.cwiseAbs().maxCoeff());
            vector dyn_residual = cur->dense_->approx_[__dyn].v_ - cur->dense_->approx_[__dyn].jac_[__y].dense() * nsp.y_y_k - cur->dense_->approx_[__dyn].jac_[__u].dense() * nsp.u_y_k;
            fmt::print("dynamics residual: {}\n", dyn_residual.cwiseAbs().maxCoeff());
            matrix dyn_jac_res = cur->dense_->approx_[__dyn].jac_[__x].dense() - cur->dense_->approx_[__dyn].jac_[__y].dense() * nsp.y_y_K - cur->dense_->approx_[__dyn].jac_[__u].dense() * nsp.u_y_K;
            fmt::print("dynamics jacobian residual: {}\n", dyn_jac_res.cwiseAbs().maxCoeff());

            if (d.ns > 0) {
                assert((d.s_y.dense() * nsp.y_y_k - cur->dense_->approx_[__eq_x].v_).cwiseAbs().maxCoeff() < 1e-10);
                assert((d.s_y.dense() * nsp.y_y_K - cur->dense_->approx_[__eq_x].jac_[__x].dense()).cwiseAbs().maxCoeff() < 1e-10);
                // fmt::print("eq x k residual {}\n", (d.s_y.dense() * nsp.y_y_k - cur->dense_->approx_[__eq_x].v_).cwiseAbs().maxCoeff());
                // fmt::print("eq x K residual {}\n", (d.s_y.dense() * nsp.y_y_K - cur->dense_->approx_[__eq_x].jac_[__x].dense()).cwiseAbs().maxCoeff());
                // fmt::print("s_y:\n{}\n", d.s_y.dense());
                // fmt::print("y_y_k:\n{}\n", nsp.y_y_k.transpose());
                // fmt::print("eq x jac:\n{}\n", cur->dense_->approx_[__eq_x].jac_[__x].dense());
            }
            if (d.nc > 0) {
                assert((d.c_u.dense() * nsp.u_y_k - cur->dense_->approx_[__eq_xu].v_).cwiseAbs().maxCoeff() < 1e-10);
                assert((d.c_u.dense() * nsp.u_y_K - cur->dense_->approx_[__eq_xu].jac_[__x].dense()).cwiseAbs().maxCoeff() < 1e-10);

                // fmt::print("eq xu k residual {}\n", (d.c_u.dense() * nsp.u_y_k - cur->dense_->approx_[__eq_xu].v_).cwiseAbs().maxCoeff());
                // fmt::print("eq xu K residual {}\n", (d.c_u.dense() * nsp.u_y_K - cur->dense_->approx_[__eq_xu].jac_[__x].dense()).cwiseAbs().maxCoeff());
            }
#endif
            d.Q_x.noalias() -= nsp.u_0_p_k.transpose() * nsp.u_y_K + nsp.u_y_k.transpose() * d.Q_ux + nsp.y_y_k.transpose() * d.Q_yx;
            d.Q_xx.noalias() -= nsp.u_0_p_K.transpose() * nsp.u_y_K + nsp.u_y_K.transpose() * d.Q_ux + nsp.y_y_K.transpose() * d.Q_yx;

            if (d.rank_status_ == rank_status::fully_constrained) {
                d.d_u.K = -nsp.u_y_K;
                d.d_u.k = -nsp.u_y_k;
                d.d_y.K = -nsp.y_y_K;
                d.d_y.k = -nsp.y_y_k;
            }
        }
    }
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto