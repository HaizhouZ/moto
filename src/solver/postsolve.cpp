#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {
namespace ns_riccati_solver {
void post_solving_steps(shooting_node *cur) {
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
    d.d_y.k.noalias() = nsp.F_0_k - nsp.F_u * d.d_u.k;
    d.d_y.K.noalias() = nsp.F_0_K - nsp.F_u * d.d_u.K;

    // compute k_lambda
    d.d_lbd_f.k.noalias() = -d.Q_y.transpose() - d.Q_yy * d.d_y.k;
    d.d_lbd_f.K.noalias() = -d.Q_xy.transpose() - d.Q_yy * d.d_y.K;
    if (d.ncstr > 0) {
        nsp.u_0_p_k.noalias() -= nsp.U * d.d_u.k; // reuse
        d.d_lbd_s_c.k = nsp.lu_eq_.solve(nsp.u_0_p_K);
        if (d.ns > 0) {
            d.d_lbd_f.k.noalias() -= nsp.s_y * d.d_lbd_s_c.k.head(d.ns);
        }
        
        nsp.u_0_p_K.noalias() -= nsp.U * d.d_u.K;
        d.d_lbd_s_c.K = nsp.lu_eq_.solve(nsp.u_0_p_K);
        if (d.ns > 0) {
            d.d_lbd_f.K.noalias() -= nsp.s_y * d.d_lbd_s_c.K.topRows(d.ns);
        }
    }
    /// @todo transpose solve
    nsp.llt_dyn_.solveInPlace(d.d_lbd_f.k);
    nsp.llt_dyn_.solveInPlace(d.d_lbd_f.K);
}
} // namespace ns_riccati_solver
} // namespace atri