#include <moto/ocp/approx_storage.hpp>
#include <moto/solver/nullspace_data.hpp>
#include <moto/solver/ns_riccati_solve.hpp>

namespace moto {
namespace ns_riccati {
void post_solving_steps(riccati_data *cur) {
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

    // compute k_y
    d.d_y.k.noalias() = -nsp.F_0_k - nsp.F_u * d.d_u.k;
    d.d_y.K.noalias() = -nsp.F_0_K - nsp.F_u * d.d_u.K;
}
} // namespace ns_riccati
} // namespace moto