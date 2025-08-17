#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <moto/solver/data_base.hpp>
#include <moto/utils/movable_ptr.hpp>

namespace moto {
struct node_data;
namespace solver {
namespace ns_riccati {
enum rank_status : int { unconstrained = 0,
                         constrained,
                         fully_constrained };

// fwd declaration
struct nullspace_data;

struct ns_node_data : public data_base {
    // dim
    size_t ns, nc, ncstr;
    size_t nis, nic; // number of active inequality constraints
    size_t nz;
    sparse_mat &F_x, &F_u;
    sparse_mat &s_y, &s_x, &c_x, &c_u;
    vector &F_0;

    movable_ptr<nullspace_data> nsp_;

    node_data *full_data_;

    rank_status rank_status_;
    // sensitivity for sqp step
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y;
    // multiplier sensitivity
    vector d_lbd_f, d_lbd_s_c_pre_solve, d_lbd_s_c;

    ns_node_data(node_data *full_data);
    ns_node_data(const ns_node_data &rhs) = delete;
    ns_node_data(ns_node_data &&rhs) = default;
    ~ns_node_data();

    void update_projected_dynamics();
    void apply_jac_y_inverse_transpose(vector &v, vector &dst);
};
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif