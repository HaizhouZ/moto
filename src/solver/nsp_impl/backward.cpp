#include <Eigen/Eigenvalues>
#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/nullspace_data.hpp>

namespace moto {
namespace nullsp_kkt_solve {
void kkt_diagnosis(riccati_data *cur) {
    auto &d = *cur;
    if (d.Q_xx.llt().info() != Eigen::Success) {
        fmt::print("Q_xx is not positive definite\n");
        fmt::print("Eigenvalues of Q_xx: \n{}\n", d.Q_xx.eigenvalues().transpose());
    }
    /// @todo some more maybe about constraints
}
void riccati_recursion(riccati_data *cur, riccati_data *prev) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // check positiveness
    // bool qyy_invertible = isPositiveDefinite(d.Q_yy);
    d.Q_yy.array() /= 2;
    /// @todo: temporary
    d.Q_yy = d.Q_yy + d.Q_yy.transpose().eval();
    if (d.Q_yy.hasNaN()) {
        fmt::print("Q_yy: \n {}\n", d.Q_yy);
        fmt::print("Q_yy has NaN\n");
    }
    nsp.U.noalias() = d.Q_uu + nsp.F_u.transpose() * d.Q_yy * nsp.F_u;
    // compute bar{u}_0
    nsp.u_0_p_k.noalias() = d.Q_u.transpose() - nsp.F_u.transpose() * (d.Q_y.transpose() - d.Q_yy * nsp.F_0_k);
    nsp.Q_yy_F_0_K.noalias() = d.Q_yy * nsp.F_0_K;
    nsp.u_0_p_K.noalias() = d.Q_ux - nsp.F_u.transpose() * (d.Q_yx - nsp.Q_yy_F_0_K);
    // compute u_y

    // fmt::print("u_0_p_k: \n{}\n", nsp.u_0_p_k.transpose());
    // fmt::print("u_0_p_K: \n{}\n", nsp.u_0_p_K.transpose());
    // fmt::print("u_y_K: \n{}\n", nsp.u_y_K.transpose());
    // compute z_u
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_u_k.noalias() = nsp.u_0_p_k;
        nsp.z_u_K.noalias() = nsp.u_0_p_K;
        d.d_u.K.noalias() = -nsp.z_u_K;
        nsp.llt_ns_.compute(nsp.U);
        if (nsp.llt_ns_.info() != Eigen::Success) {
            fmt::print("Q_uu: \n{}\n", d.Q_uu);
            kkt_diagnosis(cur);
            throw std::runtime_error("U is not positive definite");
        } else
            nsp.llt_ns_.solveInPlace(d.d_u.K);
    } else {
        // constr rank > 0
        nsp.z_u_k.noalias() = nsp.u_0_p_k - nsp.U * nsp.u_y_k;
        nsp.z_u_K.noalias() = nsp.u_0_p_K - nsp.U * nsp.u_y_K;
        if (d.rank_status_ == rank_status::fully_constrained) {
            // fully constrained
            d.d_u.K = -nsp.u_y_K;
        } else {
            // solve nullspace system
            nsp.U_z.noalias() = nsp.Z.transpose() * nsp.U * nsp.Z;
            nsp.u_z_K.noalias() = -nsp.Z.transpose() * nsp.z_u_K;
            nsp.llt_ns_.compute(nsp.U_z);
            nsp.llt_ns_.solveInPlace(nsp.u_z_K);
            d.d_u.K.noalias() = nsp.Z * nsp.u_z_K - nsp.u_y_K;
        }
    }

    // update value function derivatives of previous node
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        // update P
        d_pre.Q_y.noalias() += -d.Q_y * nsp.F_0_K + nsp.F_0_k.transpose() * nsp.Q_yy_F_0_K +
                               nsp.z_u_k.transpose() * d.d_u.K;
        d_pre.Q_yy.noalias() += nsp.F_0_K.transpose() * nsp.Q_yy_F_0_K + nsp.z_u_K.transpose() * d.d_u.K;
        if (d.ncstr > 0) {
            d.Q_y.noalias() -= nsp.u_y_k.transpose() * nsp.u_0_p_K;
            d.Q_yy.noalias() -= nsp.u_y_K.transpose() * nsp.u_0_p_K;
        }
    }
}
} // namespace nullsp_kkt_solve
} // namespace moto
