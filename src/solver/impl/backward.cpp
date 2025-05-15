#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/ns_riccati_solve.hpp>
#include <iostream>

namespace atri {
namespace ns_riccati {
// bool isPositiveDefinite(const Eigen::MatrixXd &A) {
//     if (!A.isApprox(A.transpose())) {
//         // Not symmetric
//         fmt::println("matrix non symmetric");
//         return false;
//     }
//     Eigen::LLT<Eigen::MatrixXd> llt(A);
//     return llt.info() == Eigen::Success;
// }
void backward_pass(node *cur, node *prev) {
    auto &d = get_data(cur);
    auto &nsp = *d.nsp_;
    // check positiveness
    // bool qyy_invertible = isPositiveDefinite(d.Q_yy);
    nsp.U.noalias() = d.Q_uu + nsp.F_u.transpose() * d.Q_yy * nsp.F_u;
    // bool U_invertible = isPositiveDefinite(nsp.U);
    // compute bar{u}_0
    nsp.u_0_p_k.noalias() = d.Q_u.transpose() - nsp.F_u.transpose() * (d.Q_y.transpose() - d.Q_yy * nsp.F_0_k);
    nsp.u_0_p_K.noalias() = d.Q_ux - nsp.F_u.transpose() * (d.Q_yx - d.Q_yy * nsp.F_0_K);
    // compute z_u
    if (d.rank_status_ == rank_status::unconstrained) {
        nsp.z_u_k.noalias() = nsp.u_0_p_k;
        nsp.z_u_K.noalias() = nsp.u_0_p_K;
        d.d_u.K.noalias() = -nsp.z_u_K;
        nsp.llt_ns_.compute(nsp.U);
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
            /// @todo: what if nsp.u_z_K size is wrong
            nsp.u_z_K.noalias() = -nsp.Z.transpose() * nsp.z_u_K;
            nsp.llt_ns_.compute(nsp.U_z);
            nsp.llt_ns_.solveInPlace(nsp.u_z_K);
            d.d_u.K.noalias() = nsp.Z * nsp.u_z_K - nsp.u_y_K;
        }
    }

    // update value function derivatives of previous node
    if (prev != nullptr) [[likely]] {
        auto &d_pre = get_data(prev);
        // update P
        d_pre.Q_y.noalias() += -d.Q_y * nsp.F_0_K + nsp.F_0_k.transpose() * d.Q_yy * nsp.F_0_K +
                               nsp.z_u_k.transpose() * d.d_u.K;
        d_pre.Q_yy.noalias() += nsp.F_0_K.transpose() * d.Q_yy * nsp.F_0_K + nsp.z_u_K.transpose() * d.d_u.K;
        if (d.ncstr > 0) {
            d.Q_y.noalias() -= nsp.u_y_k.transpose() * nsp.u_0_p_K;
            d.Q_yy.noalias() -= nsp.u_y_K.transpose() * nsp.u_0_p_K;
        }
    }
}
} // namespace ns_riccati
} // namespace atri
