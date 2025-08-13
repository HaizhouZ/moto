#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ns_riccati/nullspace_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void compute_primal_sensitivity(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // compute k_u
    if (d.rank_status_ == rank_status::unconstrained) {
        d.d_u.k = -nsp.z_u_k;
        nsp.llt_ns_.solveInPlace(d.d_u.k);
    } else if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.k = -nsp.u_y_k;
    } else {
        nsp.u_z_k.noalias() = -nsp.Z.transpose() * nsp.z_u_k;
        nsp.llt_ns_.solveInPlace(nsp.u_z_k);
        d.d_u.k.noalias() = nsp.Z * nsp.u_z_k - nsp.u_y_k;
    }
    auto &F_u = d.dense_->dynamics_data_.proj_f_u_;
    auto &F_x = d.dense_->dynamics_data_.proj_f_x_;
    auto &f = d.dense_->dynamics_data_.v_;
    // compute k_y
    // d.d_y.k.noalias() = -nsp.F_0_k - nsp.F_u * d.d_u.k;
    // d.d_y.K.noalias() = -nsp.F_0_K - nsp.F_u * d.d_u.K;
    d.d_y.k = -f;
    F_u.times<false>(d.d_u.k, d.d_y.k);
    F_x.dump_into(d.d_y.K, false);
    F_u.times<false>(d.d_u.K, d.d_y.K);
}
void compute_primal_sensitivity_correction(ns_node_data *cur) {
    auto &d = *cur;
    auto &nsp = *d.nsp_;
    // k_u correction
    if (d.rank_status_ == rank_status::unconstrained) {
        d.d_u.k = -nsp.z_u_k;
        nsp.llt_ns_.solveInPlace(d.d_u.k);
    } else if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.k.setZero();
    } else {
        // u_z_k correction step
        nsp.u_z_k.noalias() = -nsp.Z.transpose() * nsp.z_u_k;
        nsp.llt_ns_.solveInPlace(nsp.u_z_k);
        d.d_u.k.noalias() = nsp.Z * nsp.u_z_k;
    }

    // k_y correction
    // d.d_y.k.noalias() = -nsp.F_u * d.d_u.k;
    auto &F_u = d.dense_->dynamics_data_.proj_f_u_;
    F_u.times<false>(d.d_u.k, d.d_y.k);
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto