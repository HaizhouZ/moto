#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void generic_solver::compute_primal_sensitivity(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    // compute k_u
    if (d.rank_status_ == rank_status::unconstrained) {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_u.k = nsp.z_k;
        d.d_u.K = nsp.z_K;
        d.d_y.k = -d.F_0;
        d.F_u.times<false>(d.d_u.k, d.d_y.k);
        d.d_y.K.setZero();
        d.F_x.dump_into(d.d_y.K, spmm::dump_config{.add = false});
        d.F_u.times<false>(d.d_u.K, d.d_y.K);
    } else if (d.rank_status_ == rank_status::fully_constrained) {
    } else {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_u.k.noalias() = nsp.Z_u * nsp.z_k - nsp.u_y_k;
        if (d.d_u.k.hasNaN()) {
            fmt::print("nsp.Z_u: \n{}\n", nsp.Z_u);
            fmt::print("nsp.Z_y: \n{}\n", nsp.Z_y);
            fmt::print("nsp.z_0_k: {}\n", nsp.z_0_k.transpose());
            fmt::print("nsp.y_0_p_k: {}\n", nsp.y_0_p_k.transpose());
            fmt::print("nsp.u_0_p_k: {}\n", nsp.u_0_p_k.transpose());
            fmt::print("nsp.z_k: {}\n", nsp.z_k.transpose());
            fmt::print("nsp.u_y_k: {}\n", nsp.u_y_k.transpose());
            throw std::runtime_error("generic_solver compute_primal_sensitivity: d_u.k is NaN");
        }
        d.d_u.K.noalias() = nsp.Z_u * nsp.z_K - nsp.u_y_K;
        d.d_y.k.noalias() = nsp.Z_y * nsp.z_k - nsp.y_y_k;
        d.d_y.K.noalias() = nsp.Z_y * nsp.z_K - nsp.y_y_K;
    }
}
void generic_solver::compute_primal_sensitivity_correction(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    // k_u correction
    if (d.rank_status_ == rank_status::unconstrained) {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        // k_y correction
        // d.d_y.k.noalias() = -nsp.F_u * d.d_u.k;
        // d.d_y.k.noalias() = -d.F_u.dense() * d.d_u.k;
        d.d_u.k = nsp.z_k;
        d.d_y.k.setZero();
        d.F_u.times<false>(d.d_u.k, d.d_y.k);

    } else if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_u.k.setZero();
        d.d_y.k.setZero();
    } else {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_u.k.noalias() = nsp.Z_u * nsp.z_k; // - nsp.u_y_k;
        d.d_y.k.noalias() = nsp.Z_y * nsp.z_k; // - nsp.y_y_k;
    }
} 
} // namespace ns_riccati
} // namespace solver
} // namespace moto
