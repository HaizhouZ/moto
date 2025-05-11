#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__


#include <atri/ocp/core/node_data.hpp>

namespace atri {
enum rank_status : int { unconstrained = 0, constrained, fully_constrained };

// fwd declaration
struct nullspace_data;
struct rollout_data;

struct nullspace_riccati_data : public node_data {
    // dim
    size_t nx, nu, ns, nc, ncstr;
    size_t nz;
    // value function
    row_vector_ref Q_x, Q_u, Q_y;
    matrix_ref Q_xx, Q_xu, Q_uu, Q_xy, Q_yy;

    nullspace_data *nsp_;

    rank_status rank_status_;
    // sensitivity for sqp step
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y, d_lbd_f, d_lbd_s_c;
    // linear rollout
    rollout_data *rollout_;
    nullspace_riccati_data(problem_ptr_t prob);
    ~nullspace_riccati_data();
};
def_unique_ptr(nullspace_riccati_data);

} // namespace atri

#endif