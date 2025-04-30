#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_riccati_solver::post_solving_steps() {
    // compute zero order sensitivities and the rest 1st order ones
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        // compute k_u
        if (d.rank_status_ == rank_status::unconstrained) {
            d.d_u.k = d.z_u_k;
            d.llt_ns_.solveInPlace(d.d_u.k);
        } else if (d.rank_status_ == rank_status::fully_constrained) {
            d.d_u.k = d.u_y_k;
        } else {
            d.u_z_k.noalias() = d.Z.transpose() * d.z_u_k;
            d.llt_ns_.solveInPlace(d.u_z_k);
            d.d_u.k.noalias() = d.u_y_k + d.Z * d.u_z_k;
        }

        // compute k_y
        d.d_y.k.noalias() = d.F_0_k + d.F_u * d.d_u.k;
        d.d_y.K.noalias() = d.F_0_K + d.F_u * d.d_u.K;

        // compute k_lambda
        d.u_0_p_k.noalias() -= d.U * d.d_u.k; // reuse
        d.d_lbd_s_c.k = d.lu_eq_.solve(d.u_0_p_K);
        d.d_lbd_f.k.noalias() = -d.Q_y.transpose() - d.Q_yy * d.d_y.k -
                                d.s_y * d.d_lbd_s_c.k.head(d.ns);
        d.llt_dyn_.solveInPlace(d.d_lbd_f.k);

        d.u_0_p_K.noalias() -= d.U * d.d_u.K;
        d.d_lbd_s_c.K = d.lu_eq_.solve(d.u_0_p_K);
        d.d_lbd_f.K.noalias() = -d.Q_xy.transpose() - d.Q_yy * d.d_y.K -
                                d.s_y * d.d_lbd_s_c.K.topRows(d.ns);
        d.llt_dyn_.solveInPlace(d.d_lbd_f.K);
    }
}
} // namespace atri