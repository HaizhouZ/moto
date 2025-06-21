#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <atri/ocp/node_data.hpp>

namespace atri {
namespace ns_riccati {
enum rank_status : int { unconstrained = 0,
                         constrained,
                         fully_constrained };

// fwd declaration
struct nullspace_data;
struct rollout_data;

struct riccati_data : public node_data {
    // dim
    size_t nx, nu, ns, nc, ncstr;
    size_t nz;
    // value function
    row_vector& Q_x;
    row_vector& Q_u;
    row_vector& Q_y;
    matrix& Q_xx;
    matrix& Q_ux;
    matrix& Q_uu;
    matrix& Q_yx;
    matrix& Q_yy;

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
    // linear rollout
    rollout_data *rollout_;
    riccati_data(const problem_ptr_t& prob);
    ~riccati_data();
};
} // namespace ns_riccati
} // namespace atri

#endif