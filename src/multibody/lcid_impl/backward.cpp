#include <moto/multibody/lcid_riccati/lcid_solver.hpp>

namespace moto {
namespace solver {
namespace lcid_riccati {
void lcid_solver::riccati_recursion(ns_riccati_data *cur, ns_riccati_data *nxt) {
    auto &d = *cur;
    auto &aux = static_cast<lcid_solver::data &>(*d.aux_);
    auto &l = lcid_.as<multibody::lcid>();
    auto &nsp = d.nsp_;

    d.V_yy.array() /= 2;
    /// @todo: temporary
    d.V_yy = d.V_yy + d.V_yy.transpose().eval();
    aux.f_y.inner_product(d.V_yy, aux.Q_yy_p);
    // update Q_zz
    aux.f_z_T_Q_yy_p.setZero();
    aux.f_z.T_times(aux.Q_yy_p, aux.f_z_T_Q_yy_p);
    aux.f_z.right_times(aux.f_z_T_Q_yy_p, nsp.Q_zz);

    // update z_0_k += Z^y^T *Q_y.T - f_z^T V_yy_p * (f_0 - f_u * u_y_k)
    nsp.z_0_k.noalias() += nsp.Z_y.transpose() * d.Q_y;
    nsp.z_0_k.noalias() += aux.f_z_T_Q_yy_p * nsp.y_y_k;
    

    // update z_0_K += f_z^T * V_yy_p * (f_x - f_u * u_y_K)
    aux.f_x.right_times(aux.f_z_T_Q_yy_p, nsp.z_0_K);
    aux.f_u_times_u_y_K.right_times<false>(aux.f_z_T_Q_yy_p, nsp.z_0_K);
    nsp.llt_ns_.solve(nsp.z_0_K, nsp.z_K, -1.0);

    // update value function derivatives
        //     d.Q_x.noalias() += nsp.z_0_k.transpose() * nsp.z_K - nsp.y_0_p_k.transpose() * nsp.y_y_K;
        // d.V_xx.noalias() += nsp.z_0_K.transpose() * nsp.z_K - nsp.y_0_p_K.transpose() * nsp.y_y_K;
}
} // namespace lcid_riccati
} // namespace solver
} // namespace moto