#ifndef __NULLSPACE_DATA__
#define __NULLSPACE_DATA__

#include <moto/core/fwd.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <moto/utils/blasfeo_factorization.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
/**
 * @brief null space data struct, contains all the elements in null-space based KKT solving
 *
 */
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
    size_t rank{0};                  ///< rank of the equality constraints, 0 if unconstrained, ncstr if fully constrained
};
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif