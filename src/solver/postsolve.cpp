#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {
void nullspace_riccati_solver::post_solving_steps() {
    // compute zero order sensitivities and the rest 1st order ones
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        auto &nsp = *d.nsp_;
        // compute k_u
        if (d.rank_status_ == rank_status::unconstrained) {
            d.d_u.k = nsp.z_u_k;
            nsp.llt_ns_.solveInPlace(d.d_u.k);
        } else if (d.rank_status_ == rank_status::fully_constrained) {
            d.d_u.k = nsp.u_y_k;
        } else {
            nsp.u_z_k.noalias() = nsp.Z.transpose() * nsp.z_u_k;
            nsp.llt_ns_.solveInPlace(nsp.u_z_k);
            d.d_u.k.noalias() = nsp.u_y_k + nsp.Z * nsp.u_z_k;
        }

        // compute k_y
        d.d_y.k.noalias() = nsp.F_0_k + nsp.F_u * d.d_u.k;
        d.d_y.K.noalias() = nsp.F_0_K + nsp.F_u * d.d_u.K;

        // compute k_lambda
        nsp.u_0_p_k.noalias() -= nsp.U * d.d_u.k; // reuse
        d.d_lbd_s_c.k = nsp.lu_eq_.solve(nsp.u_0_p_K);
        d.d_lbd_f.k.noalias() = -d.Q_y.transpose() - d.Q_yy * d.d_y.k -
                                nsp.s_y * d.d_lbd_s_c.k.head(d.ns);
        nsp.llt_dyn_.solveInPlace(d.d_lbd_f.k);

        nsp.u_0_p_K.noalias() -= nsp.U * d.d_u.K;
        d.d_lbd_s_c.K = nsp.lu_eq_.solve(nsp.u_0_p_K);
        d.d_lbd_f.K.noalias() = -d.Q_xy.transpose() - d.Q_yy * d.d_y.K -
                                nsp.s_y * d.d_lbd_s_c.K.topRows(d.ns);
        nsp.llt_dyn_.solveInPlace(d.d_lbd_f.K);
    }
}
} // namespace atri