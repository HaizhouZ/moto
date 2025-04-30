#ifndef __NS_SQP__
#define __NS_SQP__

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <atri/ocp/shooting_node.hpp>
#include <list>

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
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y, d_lbd_f, d_lbd_s_c;
    matrix s_u;
    matrix F_u;
    vector F_0_k;
    matrix F_0_K;
    matrix s_c_stacked;
    vector s_c_stacked_0_k;
    matrix s_c_stacked_0_K;
    Eigen::FullPivLU<matrix> lu_;
    Eigen::LLT<matrix> llt_;
    Eigen::LLT<matrix> llt_dyn_;
    rank_status rank_status_;
    nullspace_riccati_data(expr_sets_ptr_t exprs);
};

def_ptr(nullspace_riccati_data);

class nullspace_riccati_solver {
    data_mgr &mem_;
    std::vector<shooting_node_ptr_t> nodes_;

  public:
    nullspace_riccati_solver()
        : mem_(data_mgr::get<nullspace_riccati_data>()) {}
    void set_horizon(size_t N) { nodes_.resize(N); }
    static auto &get_data(shooting_node_ptr_t node) {
        return *std::static_pointer_cast<nullspace_riccati_data>(node->data_);
    }

  private:
    void pre_solving_steps();
    void backward_pass();
    void post_solving_steps();
    void forward_rollout();
    void post_rollout_steps();
};

} // namespace atri

#endif