#include <atri/solver/ns_riccati_solver.hpp>

#include <atri/solver/data/nullspace_data.hpp>

namespace atri {

void nullspace_riccati_solver::backward_pass() {
    /// @todo check set terminal should be in cost computation
    for (size_t i = nodes_.size(); i >= 0; i--) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        auto &nsp = *d.nsp_;
        nsp.U.noalias() = d.Q_uu + nsp.F_u.transpose() * d.Q_yy * nsp.F_u;
        // compute bar{u}_0
        nsp.u_0_p_k.noalias() =
            -d.Q_u.transpose() -
            nsp.F_u.transpose() * (-d.Q_y.transpose() - d.Q_yy * nsp.F_0_k);
        nsp.u_0_p_K.noalias() =
            -d.Q_xu.transpose() -
            nsp.F_u.transpose() * (-d.Q_xy.transpose() - d.Q_yy * nsp.F_0_K);
        // compute z_u
        if (d.rank_status_ == rank_status::unconstrained) {
            nsp.z_u_k = nsp.u_0_p_k;
            nsp.z_u_K = nsp.u_0_p_K;
            d.d_u.K = nsp.z_u_K;
            nsp.llt_ns_.compute(nsp.U);
            nsp.llt_ns_.solveInPlace(d.d_u.K);
        } else {
            // constr rank > 0
            nsp.z_u_k.noalias() = nsp.u_0_p_k - nsp.U * nsp.u_y_k;
            nsp.z_u_K.noalias() = nsp.u_0_p_K - nsp.U * nsp.u_y_K;
            if (d.rank_status_ == rank_status::fully_constrained) {
                // fully constrained
                d.d_u.K = nsp.u_y_K;
            } else {
                // solve nullspace system
                nsp.U_z.noalias() = nsp.Z.transpose() * nsp.U * nsp.Z;
                /// @todo: what if nsp.u_z_K size is wrong
                nsp.u_z_K.noalias() = nsp.Z.transpose() * nsp.z_u_K;
                nsp.llt_ns_.compute(nsp.U_z);
                nsp.llt_ns_.solveInPlace(nsp.u_z_K);
                d.d_u.K.noalias() = nsp.u_y_K + nsp.Z * nsp.u_z_K;
            }
        }

        // update value function derivatives of previous node
        if (i > 0) {
            auto pre = nodes_[i - 1];
            auto &d_pre = get_data(pre);
            // update P
            d_pre.Q_y.noalias() +=
                -d.Q_y * nsp.F_0_K - nsp.F_0_k.transpose() * d.Q_yy * nsp.F_0_K;
            d_pre.Q_yy.noalias() -= nsp.F_0_K.transpose() * d.Q_yy * nsp.F_0_K;
            // update S
            d_pre.Q_y.noalias() += nsp.z_u_k.transpose() * d.d_u.K;
            d_pre.Q_yy.noalias() += nsp.z_u_K.transpose() * d.d_u.K;
            if (d.nc + d.ns > 0) {
                d.Q_y.noalias() += nsp.u_y_k.transpose() * nsp.u_0_p_K;
                d.Q_yy.noalias() += nsp.u_y_K.transpose() * nsp.u_0_p_K;
            }
        }
    }
}
} // namespace atri
