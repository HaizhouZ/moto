#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <moto/solver/data_base.hpp>

namespace moto {
namespace ns_riccati {
enum rank_status : int { unconstrained = 0,
                         constrained,
                         fully_constrained };

// fwd declaration
struct nullspace_data;

struct ns_node_data : public solver::data_base {
    // dim
    size_t ns, nc, ncstr;
    size_t nz;

    nullspace_data *nsp_;

    rank_status rank_status_;
    // sensitivity for sqp step
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y;
    // multiplier sensitivity
    vector d_lbd_f, d_lbd_s_c_pre_solve, d_lbd_s_c;

    shifted_array<vector, std::size(solver::hard_constr_fields), __dyn> dual_step; // dual rollout

    ns_node_data(const ocp_ptr_t &prob);
    ~ns_node_data();
};
} // namespace ns_riccati
} // namespace moto

#endif