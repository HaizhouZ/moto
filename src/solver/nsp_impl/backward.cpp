#include <Eigen/Eigenvalues>
#include <moto/ocp/dynamics.hpp>
#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/nullspace_data.hpp>

namespace moto {
namespace nullsp_kkt_solve {
void kkt_diagnosis(riccati_data *cur) {
    auto &d = *cur;
    fmt::print("U is not positive definite\n");
    fmt::print("Eigenvalues of U: \n{}\n", d.nsp_->U.eigenvalues().transpose());
    fmt::print("Eigenvalues of Q_yy: \n{}\n", d.Q_yy.eigenvalues().transpose());
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
    d.Q_yy.diagonal().array() += 1e-6; // ensure positive definiteness
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
            fmt::print("Q_yy: \n{}\n", d.Q_yy);
            fmt::print("U: \n{}\n", nsp.U);
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
        } else {
            // solve nullspace system
            nsp.U_z.noalias() = nsp.Z.transpose() * nsp.U * nsp.Z;
            nsp.u_z_K.noalias() = -nsp.Z.transpose() * nsp.z_u_K;
            nsp.llt_ns_.compute(nsp.U_z);
            nsp.llt_ns_.solveInPlace(nsp.u_z_K);
            d.d_u.K.noalias() = nsp.Z * nsp.u_z_K - nsp.u_y_K;
        }
    }
    d.Q_x.noalias() += -d.Q_y * nsp.F_0_K + nsp.F_0_k.transpose() * nsp.Q_yy_F_0_K +
                       nsp.z_u_k.transpose() * d.d_u.K;
    d.Q_xx.noalias() += nsp.F_0_K.transpose() * nsp.Q_yy_F_0_K + nsp.z_u_K.transpose() * d.d_u.K;
    if (d.rank_status_ != rank_status::unconstrained) {
        d.Q_x.noalias() -= nsp.u_y_k.transpose() * nsp.u_0_p_K;
        d.Q_xx.noalias() -= nsp.u_y_K.transpose() * nsp.u_0_p_K;
    }
    // update value function derivatives of previous node
    if (prev != nullptr) [[likely]] {
        auto &d_pre = *prev;
        auto& perm = permutation_from_y_to_x(prev->ocp_, cur->ocp_);
        d.Q_x *= perm;
        d.Q_xx *= perm;
        d.Q_xx.applyOnTheLeft(perm.transpose());
        d_pre.Q_y.noalias() += d.Q_x;
        d_pre.Q_yy.noalias() += d.Q_xx;
    }
}
} // namespace nullsp_kkt_solve
} // namespace moto
