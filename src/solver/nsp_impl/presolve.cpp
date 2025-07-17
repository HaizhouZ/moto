#include <Eigen/Eigenvalues>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace ns_riccati {

void update_approx(ns_node_data *cur) {
    // collect constraint residuals and jacobians
    auto &d = *cur;
    d.Q_x.setZero();
    d.Q_u.setZero();
    d.Q_y.setZero();
    d.Q_xx.setZero();
    d.Q_ux.setZero();
    d.Q_uu.setZero();
    d.Q_yx.setZero();
    d.Q_yy.setZero();
    // update everything
    cur->update_approximation();
}

void ns_factorization(ns_node_data *cur) {
    // collect constraint residuals and jacobians
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    auto &_approx = d.dense_->approx_;
    /// @todo sparse F_y inverse
    auto &F = _approx[__dyn].jac_;
    nsp.lu_dyn_.compute(F[__y]);
    nsp.F_u = nsp.lu_dyn_.solve(F[__u]);
    nsp.F_0_k = nsp.lu_dyn_.solve(_approx[__dyn].v_);
    nsp.F_0_K = nsp.lu_dyn_.solve(F[__x]);
    // partial value derivative
    d.Q_x.noalias() += -nsp.F_0_k.transpose() * d.Q_yx;
    if (nsp.F_0_K.array().isNaN().any() || d.Q_xx.array().isNaN().any()) {
        std::cerr << "F_0_K:\n"
                  << nsp.F_0_K << "\n";
        std::cerr << "Q_xx:\n"
                  << d.Q_xx << "\n";
        throw std::runtime_error("NaN detected in F_0_K or Q_xx");
    }
    d.Q_xx.noalias() += -(d.Q_yx.transpose() * nsp.F_0_K + nsp.F_0_K.transpose() * d.Q_yx);
    // nullspace computation
    if (d.ncstr) {
        if (d.ns) {
            nsp.s_u.noalias() = -nsp.s_y * nsp.F_u;
            // solve pseudo inverse
            nsp.s_c_stacked.topRows(d.ns) = nsp.s_u;
        }
        if (d.nc) {
            nsp.s_c_stacked.bottomRows(d.nc) = _approx[__eq_xu].jac_[__u];
        }
        nsp.lu_eq_.compute(nsp.s_c_stacked);
        size_t rank = nsp.lu_eq_.rank();
        // fmt::print("rank of equality constraints: {}\n", rank);

        if (rank == 0)
            d.rank_status_ = rank_status::unconstrained;
        else {
            if (rank == d.ncstr) {
                d.rank_status_ = rank_status::fully_constrained;
            } else {
                nsp.Z = nsp.lu_eq_.kernel();
                // fmt::print("nullspace :\n {}\n", nsp.Z);
                d.rank_status_ = rank_status::constrained;
                nsp.U_z.conservativeResize(nsp.Z.cols(), nsp.Z.cols());
                nsp.u_z_k.conservativeResize(nsp.Z.cols());
                nsp.u_z_K.conservativeResize(nsp.Z.cols(), Eigen::NoChange);
            }
        }
        // precompute
        if (d.ns) {
            nsp.s_0_p_k.noalias() =
                _approx[__eq_x].v_ - nsp.s_y * nsp.F_0_k;
            nsp.s_0_p_K.noalias() =
                _approx[__eq_x].jac_[__x] - nsp.s_y * nsp.F_0_K;
            nsp.s_c_stacked_0_k.head(d.ns) = nsp.s_0_p_k;
            nsp.s_c_stacked_0_K.topRows(d.ns) = nsp.s_0_p_K;
        }
        if (d.nc) {
            nsp.s_c_stacked_0_k.tail(d.nc) = _approx[__eq_xu].v_;
            nsp.s_c_stacked_0_K.bottomRows(d.nc) = _approx[__eq_xu].jac_[__x];
        }
        if (d.rank_status_ != rank_status::unconstrained) {
            // pre compute
            nsp.u_y_k.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_k);
            nsp.u_y_K.noalias() = nsp.lu_eq_.solve(nsp.s_c_stacked_0_K);
            // fmt::print("jac: \n {}\n", nsp.s_c_stacked);
            // fmt::print("scstacked_0_k :\n {}\n", nsp.s_c_stacked_0_k.transpose());
            // fmt::print("nyk :\n {}\n", nsp.u_y_k.transpose());
            // fmt::print("scstacked_0_K :\n {}\n", nsp.s_c_stacked_0_K);
            // fmt::print("nyK :\n {}\n", nsp.u_y_K);
            // fmt::print("F_u: \n{}\n", nsp.F_u.transpose());
            // fmt::print("F_0_K: \n{}\n", nsp.F_0_K.transpose());
            if (d.rank_status_ == rank_status::fully_constrained) {
                d.d_u.K = -nsp.u_y_K;
            }
        }
    }
}

} // namespace ns_riccati
} // namespace moto