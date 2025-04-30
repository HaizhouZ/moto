#ifndef __NULLSPACE_DATA__
#define __NULLSPACE_DATA__

#include <atri/core/fwd.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace atri {
struct nullspace_data {
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
    nullspace_data(matrix_ref _s_y) : s_y(_s_y) {}
};
} // namespace atri

#endif