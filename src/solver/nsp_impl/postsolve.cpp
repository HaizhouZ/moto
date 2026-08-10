#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/core/linear_backend.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
namespace {
void ensure_nullspace_factorization(ns_riccati_data &d) {
    auto &nsp = d.nsp_;
    const auto factor_rows = static_cast<Eigen::Index>(nsp.llt_ns_.L_.data_.m);
    const auto factor_cols = static_cast<Eigen::Index>(nsp.llt_ns_.L_.data_.n);
    if (factor_rows != nsp.Q_zz.rows() || factor_cols != nsp.Q_zz.cols()) {
        nsp.llt_ns_.compute(nsp.Q_zz);
    }
}

} // namespace

void generic_solver::compute_primal_sensitivity(ns_riccati_data *cur) {
    auto &d = *cur;
    auto &nsp = d.nsp_;
    // compute k_u
    if (d.rank_status_ == rank_status::unconstrained) {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        ensure_nullspace_factorization(d);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_y.k = -d.F_0;
        linear_backend::multiply(d.F_u, nsp.z_k, d.d_y.k, -1.);
        d.d_y.K.setZero();
        linear_backend::write_dense(d.F_x, d.d_y.K, {.alpha = -1.});
        linear_backend::multiply(d.F_u, nsp.z_K, d.d_y.K, -1.);
    } else if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_y.k = -nsp.y_y_k;
        d.d_y.K = -nsp.y_y_K;
    } else {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        ensure_nullspace_factorization(d);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
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
        ensure_nullspace_factorization(d);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_y.k.setZero();
        linear_backend::multiply(d.F_u, nsp.z_k, d.d_y.k, -1.);

    } else if (d.rank_status_ == rank_status::fully_constrained) {
        d.d_y.k.setZero();
    } else {
        // nsp.z_k = -nsp.z_0_k;
        // nsp.llt_ns_.solveInPlace(nsp.z_k);
        ensure_nullspace_factorization(d);
        nsp.llt_ns_.solve(nsp.z_0_k, nsp.z_k, -1.0);
        d.d_y.k.noalias() = nsp.Z_y * nsp.z_k; // - nsp.y_y_k;
    }
} 
} // namespace ns_riccati
} // namespace solver
} // namespace moto
