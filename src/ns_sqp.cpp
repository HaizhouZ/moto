#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_riccati_solver::backward_pass() {
    /// @todo set terminal
    for (size_t k = nodes_.size(); k > 0; k--) {
        auto next = nodes_[k];
        auto &dnext =
            *std::static_pointer_cast<nullspace_riccati_data>(next->data_);
        auto cur = nodes_[k - 1];
        auto &d = *std::static_pointer_cast<nullspace_riccati_data>(cur->data_);
        // update value function derivatives
        d.value_func_.Q_y += dnext.value_func_.V_y;
        d.value_func_.Q_yy += dnext.value_func_.V_yy;
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
        auto Z = d.qr_Q.rightCols(d.nz);
        d.U_z = Z.transpose() * d.U * Z;
        d.llt_.compute(d.U_z);
    }
}
} // namespace atri
