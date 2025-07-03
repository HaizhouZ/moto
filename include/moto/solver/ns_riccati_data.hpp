#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <moto/ocp/node_data.hpp>
#include <moto/solver/solver_data.hpp>

namespace moto {
namespace ns_riccati {
enum rank_status : int { unconstrained = 0,
                         constrained,
                         fully_constrained };

// fwd declaration
struct nullspace_data;

constexpr field_t hard_constr_fields[] = {__dyn, __eq_x, __eq_xu};

struct riccati_data : public node_data, public solver::solver_data {
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

    shifted_array<vector, std::size(hard_constr_fields), __dyn> dual_rollout_; // dual rollout

    riccati_data(const ocp_ptr_t &prob);
    ~riccati_data();
};
} // namespace ns_riccati
} // namespace moto

#endif