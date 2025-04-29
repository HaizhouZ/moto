#ifndef __NS_SQP__
#define __NS_SQP__

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <atri/ocp/shooting_node.hpp>
#include <list>

namespace atri {
struct value_func_data {
    row_vector Q_x, Q_u;
    matrix Q_xx, Q_ux, Q_uu;
    row_vector Q_y;
    matrix Q_yy;

    value_func_data(size_t nx, size_t nu)
        : Q_x(nx), Q_u(nu), Q_xx(nx, nx), Q_ux(nu, nx), Q_uu(nu, nu), Q_y(nx),
          Q_yy(nx, nx) {}
};

struct nullspace_riccati_data : public node_data {
    value_func_data value_func_;
    matrix U;       // projected u hessian
    matrix U_z;     // nullspace u hessian
    matrix Z;       // nullspace basis
    vector z_u_k;   // z_u residual [\bar[u]_0 - U \delta u_y]
    vector u_y_k;   // u_y psuedo u
    vector u_z_k;   // u_z nullspace coefficient
    vector u_0_p_k; // u_0 projected
    matrix z_u_K;
    matrix u_y_K;
    matrix u_z_K;
    matrix u_0_p_K;
    vector k_u;
    matrix K_u;
    matrix F_u;
    vector u_0_k;
    vector y_0_k;
    vector F_0_k;
    matrix u_0_K;
    matrix y_0_K;
    matrix F_0_K;
    matrix P_k;
    matrix P_K;
    matrix S_k;
    matrix S_K;
    Eigen::FullPivLU<matrix> lu_;
    Eigen::LLT<matrix> llt_;
    size_t nx, nu;
    size_t nz;
    nullspace_riccati_data(expr_sets_ptr_t exprs)
        : node_data(exprs), nx(exprs->dim_[field::x]),
          nu(exprs->dim_[field::u]), value_func_(nx, nu), K_u(nu, nx), k_u(nu),
          F_u(nx, nu) {
        size_t rank = raw_data_.exprs_->dim_[field::eq_constr_c] +
                      raw_data_.exprs_->dim_[field::eq_constr_s];
        nz = nu - rank;
        z_u_k.resize(nu);
        z_u_K.resize(nu, nx);
        U_z.resize(nz, nz);
    }
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