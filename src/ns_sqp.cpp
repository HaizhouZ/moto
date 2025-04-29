#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_riccati_solver::pre_solving_steps() {
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        // collect constraint residuals and jacobians
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        // update everything
        cur->update_approximation();
    }
}
void nullspace_riccati_solver::backward_pass() {
    /// @todo check set terminal should be in cost computation
    for (size_t i = nodes_.size(); i >= 0; i--) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        d.U.noalias() =
            d.value_func_.Q_uu + d.F_u.transpose() * d.value_func_.Q_yy * d.F_u;
        // compute bar{u}_0
        d.u_0_p_k.noalias() =
            d.u_0_k -
            d.F_u.transpose() * (d.y_0_k - d.value_func_.Q_yy * d.F_0_k);
        d.u_0_p_K.noalias() =
            d.u_0_K -
            d.F_u.transpose() * (d.y_0_K - d.value_func_.Q_yy * d.F_0_K);
        // compute z_u
        d.z_u_k.noalias() = d.u_0_p_k - d.U * d.u_y_k;
        d.z_u_K.noalias() = d.u_0_p_K - d.U * d.u_y_K;
        // solve nullspace system
        auto &Z = d.lu_.kernel();
        d.U_z.noalias() = Z.transpose() * d.U * Z;
        d.u_z_K.noalias() = Z.transpose() * d.z_u_K;
        d.llt_.compute(d.U_z);
        d.llt_.solveInPlace(d.u_z_K);

        // update value function derivatives of previous node
        if (i > 0) {
            auto pre = nodes_[i - 1];
            auto &d_pre = get_data(pre);
            // update P
            d.value_func_.Q_y.noalias() +=
                d.F_0_k.transpose() * d.value_func_.Q_yy * d.F_0_K;
            d.value_func_.Q_yy.noalias() +=
                d.F_0_K.transpose() * d.value_func_.Q_yy * d.F_0_K;
            // update S
            d.value_func_.Q_y.noalias() =
                d.z_u_k.transpose() * d.K_u + d.u_y_k.transpose() * d.u_0_K;
            d.value_func_.Q_yy.noalias() =
                d.z_u_K.transpose() * d.K_u + d.u_y_K.transpose() * d.u_0_K;
        }
    }
}
} // namespace atri
