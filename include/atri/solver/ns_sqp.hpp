#ifndef __NS_SQP__
#define __NS_SQP__

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <atri/ocp/shooting_node.hpp>
#include <list>

namespace atri {
/// @todo replace with a graph
// using ocp_graph = std::vector<shooting_node_ptr_t>;

// class solver_base {
//   protected:
//     std::shared_ptr<ocp_graph> graph_;

//   public:
//     virtual ~solver_base() = default;

//     solver_base(std::shared_ptr<ocp_graph> graph) : graph_(graph) {}

//     virtual void backward_pass() {}
//     // update primal variables
//     virtual void forward_pass() {}

//     // update the derivatives of the primal variables
//     void update_derivatives() {
// #pragma omp parallel for
//         for (int i = 0; i < graph_->size(); i++) {
//             auto s = graph_->at(i);
//             s->update_approximation();
//         }
//     }
// };

struct value_func_data {
    row_vector Q_x, Q_u;
    matrix Q_xx, Q_ux, Q_uu;
    matrix Q_yy, V_yy;
    row_vector Q_y, V_y;

    value_func_data(size_t nx, size_t nu)
        : Q_x(nx), Q_u(nu), Q_xx(nx, nx), Q_ux(nu, nx), Q_uu(nu, nu), V_y(nx),
          V_yy(nx, nx), Q_yy(nx, nx), Q_y(nx) {}
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
    Eigen::HouseholderQR<matrix> qr_;
    matrix qr_Q;
    Eigen::LLT<matrix> llt_;
    size_t nx, nu;
    size_t nz;
    nullspace_riccati_data(expr_sets_ptr_t exprs)
        : node_data(exprs), nx(exprs->dim_[field::x]),
          nu(exprs->dim_[field::u]), value_func_(nx, nu), K_u(nu, nx), k_u(nu),
          F_u(nx, nu), qr_Q(nu, nu) {
        size_t rank = primal_data_.exprs_->dim_[field::eq_constr_c] +
                      primal_data_.exprs_->dim_[field::eq_constr_s];
        nz = nu - rank;
        z_u_k.resize(nz);
        z_u_K.resize(nz, nx);
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

  private:
    void pre_solving_steps();
    void backward_pass();
    void post_solving_steps();
    void forward_rollout();
    void post_rollout_steps();
};

} // namespace atri

#endif