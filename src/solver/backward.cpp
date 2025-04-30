#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {

void nullspace_riccati_solver::backward_pass() {
    /// @todo check set terminal should be in cost computation
    for (size_t i = nodes_.size(); i >= 0; i--) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        d.U.noalias() = d.Q_uu + d.F_u.transpose() * d.Q_yy * d.F_u;
        // compute bar{u}_0
        d.u_0_p_k.noalias() =
            -d.Q_u.transpose() -
            d.F_u.transpose() * (-d.Q_y.transpose() - d.Q_yy * d.F_0_k);
        d.u_0_p_K.noalias() =
            -d.Q_xu.transpose() -
            d.F_u.transpose() * (-d.Q_xy.transpose() - d.Q_yy * d.F_0_K);
        // compute z_u
        if (d.rank_status_ == rank_status::unconstrained) {
            d.z_u_k = d.u_0_p_k;
            d.z_u_K = d.u_0_p_K;
            d.d_u.K = d.z_u_K;
            d.llt_ns_.compute(d.U);
            d.llt_ns_.solveInPlace(d.d_u.K);
        } else {
            // constr rank > 0
            d.z_u_k.noalias() = d.u_0_p_k - d.U * d.u_y_k;
            d.z_u_K.noalias() = d.u_0_p_K - d.U * d.u_y_K;
            if (d.rank_status_ == rank_status::fully_constrained) {
                // fully constrained
                d.d_u.K = d.u_y_K;
            } else {
                // solve nullspace system
                d.U_z.noalias() = d.Z.transpose() * d.U * d.Z;
                /// todo: what if d.u_z_K size is wrong
                d.u_z_K.noalias() = d.Z.transpose() * d.z_u_K;
                d.llt_ns_.compute(d.U_z);
                d.llt_ns_.solveInPlace(d.u_z_K);
                d.d_u.K.noalias() = d.u_y_K + d.Z * d.u_z_K;
            }
        }

        // update value function derivatives of previous node
        if (i > 0) {
            auto pre = nodes_[i - 1];
            auto &d_pre = get_data(pre);
            // update P
            d_pre.Q_y.noalias() +=
                -d.Q_y * d.F_0_K - d.F_0_k.transpose() * d.Q_yy * d.F_0_K;
            d_pre.Q_yy.noalias() -= d.F_0_K.transpose() * d.Q_yy * d.F_0_K;
            // update S
            d_pre.Q_y.noalias() += d.z_u_k.transpose() * d.d_u.K;
            d_pre.Q_yy.noalias() += d.z_u_K.transpose() * d.d_u.K;
            if (d.nc + d.ns > 0) {
                d.Q_y.noalias() += d.u_y_k.transpose() * d.u_0_p_K;
                d.Q_yy.noalias() += d.u_y_K.transpose() * d.u_0_p_K;
            }
        }
    }
}
} // namespace atri
