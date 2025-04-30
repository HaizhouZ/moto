#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <atri/ocp/node_data.hpp>

namespace atri {
enum rank_status : int { unconstrained = 0, constrained, fully_constrained };

struct nullspace_riccati_data : public node_data {
    // dim
    size_t nx, nu, ns, nc, ncstr;
    size_t nz;
    // value function
    row_vector_ref Q_x, Q_u, Q_y;
    matrix_ref Q_xx, Q_xu, Q_uu, Q_xy, Q_yy;
    // jacobian
    matrix_ref s_y;
    // nullspace
    matrix U;       // projected u hessian
    matrix U_z;     // nullspace u hessian
    matrix Z;       // nullspace basis
    vector z_u_k;   // z_u residual [\bar[u]_0 - U \delta u_y]
    vector u_y_k;   // u_y psuedo u
    vector u_z_k;   // u_z nullspace coefficient
    vector u_0_p_k; // u_0 projected
    vector s_0_p_k; // s_0 - s_y F 0
    matrix z_u_K;
    matrix u_y_K;
    matrix u_z_K;
    matrix u_0_p_K;
    vector s_0_p_K; // s_0 - s_y F 0
    matrix s_u;
    matrix F_u;
    vector F_0_k;
    matrix F_0_K;
    matrix s_c_stacked;
    vector s_c_stacked_0_k;
    matrix s_c_stacked_0_K;
    Eigen::FullPivLU<matrix> lu_eq_;
    Eigen::LLT<matrix> llt_ns_;
    Eigen::LLT<matrix> llt_dyn_;
    rank_status rank_status_;
    // sensitivity for sqp step
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y, d_lbd_f, d_lbd_s_c;
    // linear rollout
    struct rollout_data {
        vector prim_[field::num_sym];
        vector dual_[field::num_constr]; // exclude cost
    } rollout_;
    nullspace_riccati_data(problem_ptr_t exprs);
};
def_ptr(nullspace_riccati_data);

} // namespace atri

#endif