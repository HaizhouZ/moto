#ifndef MOTO_SOLVER_LCID_RICCATI_LCID_SOLVER_HPP
#define MOTO_SOLVER_LCID_RICCATI_LCID_SOLVER_HPP

#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/spmm/sparse_mat.hpp>
#include <moto/multibody/lcid.hpp>
#include <moto/multibody/stacked_euler.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
struct ns_riccati_data;
}
namespace lcid_riccati {

/// @brief lifted contact inverse dynamics riccati solver
struct lcid_solver : ns_riccati::generic_solver {
    using base = ns_riccati::generic_solver;
    using ns_riccati_data = ns_riccati::ns_riccati_data;
    ns_riccati_data *create_data(node_data *full_data) override;

    custom_func lcid_;
    constr euler_;
    using func_type = multibody::lcid;
    using int_type = multibody::stacked_euler;

    void ns_factorization(ns_riccati_data *cur) override;
    void riccati_recursion(ns_riccati_data *cur, ns_riccati_data *nxt) override;
    struct data : public ns_riccati_data::aux_data {
        vector sp_u_y_k;             // [a_y; f_y; tq_y]_k
        sparse_mat sp_u_y_K;         // [a_y; f_y; tq_y]_K
        aligned_map_t sp_u_y_K_a_f;
        vector sp_y_y_k;             // f_x - f_u * u_y_k
        sparse_mat f_u_times_u_y_K; // f_u * u_y_K
        aligned_map_t f_a_times_u_y_K; // f_a  * u_y_K_a_f
        sparse_mat sp_Z_u;           // [Z_a; Z_f; Z_tq]
        aligned_map_t sp_Z_u_a, sp_Z_u_f;
        sparse_mat &f_x, &f_u, &f_y; // ref to unprojected dynamics derivatives
        sparse_mat f_z;              // f_z = f_u * Z_u
        aligned_map_t f_z_a, f_z_dt;
        matrix f_z_T_Q_yy_p;     // f_z^T * Q_yy projected
        matrix Q_yy_p;   // Q_yy projected
        size_t nz;
        data(ns_riccati_data &d, lcid_solver& solver);
    };
    
    lcid_solver(const func_type &l, const int_type& e) : lcid_(l), euler_(e) {}
};

} // namespace lcid_riccati

} // namespace solver
} // namespace moto

#endif