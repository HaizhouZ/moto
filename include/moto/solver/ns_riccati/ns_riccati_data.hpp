#ifndef __NS_RICCATI_DATA__
#define __NS_RICCATI_DATA__

#include <moto/solver/data_base.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <moto/utils/blasfeo_factorizer/blasfeo_llt.hpp>


namespace moto {
struct node_data;
namespace solver {
namespace ns_riccati {
enum rank_status : int { unconstrained = 0,
                         constrained,
                         fully_constrained };

// fwd declaration
struct nullspace_data;

struct MOTO_ALIGN_NO_SHARING ns_riccati_data : public data_base {
    // dim
    size_t ns, nc, ncstr;
    size_t nis, nic; // number of active inequality constraints
    size_t nz;
    sparse_mat &F_x, &F_u;
    sparse_mat &s_y, &s_x, &c_x, &c_u;
    vector &F_0;

    ///      @brief null space data struct, contains all the elements in null-space based KKT solving
    struct nullspace_data {
        // nullspace
        matrix U;     ///< projected u hessian
        matrix Q_zz;  ///< nullspace u hessian
        matrix Z_u;   ///< nullspace basis for u
        matrix Z_y;   ///< nullspace basis for y
        vector z_0_k; ///< residual\f$ z_u = \bar{u}_0 - U \delta u_y \f$
        vector z_k;   ///< \f$u_z\f$ nullspace solution
        matrix z_0_K; ///< sa as @ref z_u_k
        matrix z_K;   ///< same as @ref z_k
        matrix s_u;   ///< \f$s_yF_u\f$
        // matrix F_u;            ///< \f$f_y^{-1}f_u\f$
        matrix s_c_stacked; ///< \f$[s_u;c_u]\f$
        vector u_0_p_k;     ///< \f$u_0\f$ projected
        matrix u_0_p_K;     ///< same meas @ref u_0_p_k
        vector y_0_p_k;     ///< \f$y_0\f$ projected
        matrix y_0_p_K;     ///< same meas @ref y_0_p_k
        vector s_0_p_k;     ///< \f$s_0\f$ projected
        matrix s_0_p_K;     ///< same meas @ref s_0_p_k
        // vector F_0_k;          ///< \f$s_yf\f$
        // matrix F_0_K;          ///< \f$s_yf_x\f$
        vector s_c_stacked_0_k;          ///< \f$[s;c]\f$
        matrix s_c_stacked_0_K;          ///< \f$[s_x;c_x]\f$
        vector u_y_k;                    ///< \f$u_y\f$ pseudo u
        matrix u_y_K;                    ///< same as @ref u_y_k
        vector y_y_k;                    ///< \f$y_y\f$ pseudo y
        matrix y_y_K;                    ///< same as @ref y_y_k
        Eigen::FullPivLU<matrix> lu_eq_; ///< LU factorizer of the eq constraints
        // Eigen::LLT<matrix> llt_ns_;      ///< LLT solver of the projected hessian
        utils::blasfeo_llt llt_ns_; ///< LLT solver of the projected hessian
        size_t rank{0};             ///< rank of the equality constraints, 0 if unconstrained, ncstr if fully constrained
    } nsp_;

    node_data *full_data_;

    struct aux_data {
        virtual ~aux_data() = default;
    };
    std::unique_ptr<aux_data> aux_; // auxiliary data pointer, can be used to store custom data

    struct restoration_aux_data : aux_data {
        scalar_t rho_eq = 1000.0;
        scalar_t rho_ineq = 1000.0;
        scalar_t rho_u = 1e-4;
        scalar_t rho_y = 1e-4;
        scalar_t lambda_reg = 1e-8;
        scalar_t mu_bar = 1.0;
        bool initialized = false;
        bool verbose = false;
        bool outer_dual_backup_valid = false;

        resto_elastic_constr elastic_eq;
        resto_ineq_constr elastic_ineq;
        array_type<vector, constr_fields> outer_dual_backup;
        vector outer_cost_backup;
        vector g_current;
        vector g_step;
        vector u_ref;
        vector y_ref;
        vector sigma_u_sq;
        vector sigma_y_sq;
    };

    rank_status rank_status_;
    // sensitivity for sqp step
    struct sensitivity {
        vector k;
        matrix K;
        sensitivity(size_t n, size_t nx) : k(n), K(n, nx) {}
    } d_u, d_y;
    // multiplier sensitivity
    vector d_lbd_f, d_lbd_s_c_pre_solve, d_lbd_s_c;

    ns_riccati_data(node_data *full_data);
    ns_riccati_data(const ns_riccati_data &rhs) = delete;
    ns_riccati_data(ns_riccati_data &&rhs) = default;

    void update_projected_dynamics();
    void update_projected_dynamics_residual();
    void apply_jac_y_inverse_transpose(vector &v, vector &dst);
};
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif
