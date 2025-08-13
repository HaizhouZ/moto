#ifndef __NULLSPACE_DATA__
#define __NULLSPACE_DATA__

#include <moto/core/fwd.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace moto {
namespace solver {
namespace ns_riccati {
/**
 * @brief null space data struct, contains all the elements in null-space based KKT solving
 *
 */
struct nullspace_data {
    // nullspace
    matrix U;              ///< projected u hessian
    matrix U_z;            ///< nullspace u hessian
    matrix Z;              ///< nullspace basis
    vector z_u_k;          ///< residual\f$ z_u = \bar{u}_0 - U \delta u_y \f$
    vector u_z_k;          ///< \f$u_z\f$ nullspace coefficient
    matrix z_u_K;          ///< sa as @ref z_u_k
    matrix u_z_K;          ///< same as @ref u_z_k
    matrix s_u;            ///< \f$s_yF_u\f$
    // matrix F_u;            ///< \f$f_y^{-1}f_u\f$
    matrix s_c_stacked; ///< \f$[s_u;c_u]\f$
    vector u_0_p_k;        ///< \f$u_0\f$ projected
    matrix u_0_p_K;        ///< same meas @ref u_0_p_k
    vector s_0_p_k;        ///< \f$s_0 - s_y F 0\f$
    matrix s_0_p_K;        ///< same as @ref s_0_p_k
    // vector F_0_k;          ///< \f$s_yf\f$
    // matrix F_0_K;          ///< \f$s_yf_x\f$
    vector Q_yy_F_0_k;
    matrix Q_yy_F_0_K;
    vector s_c_stacked_0_k;              ///< \f$[s;c]\f$
    matrix s_c_stacked_0_K;           ///< \f$[s_x;c_x]\f$
    vector u_y_k;                        ///< \f$u_y\f$ psuedo u
    matrix u_y_K;                        ///< same as @ref u_y_k
    Eigen::FullPivLU<matrix> lu_eq_;     ///< LU factorizer of the eq constraints
    Eigen::LLT<matrix> llt_ns_;          ///< LLT solver of the projected hessian
    Eigen::PartialPivLU<matrix> lu_dyn_; ///< LU solver of the dynamics derivative \f$f_y\f$
    size_t rank{0};                      ///< rank of the equality constraints, 0 if unconstrained, ncstr if fully constrained
};
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif