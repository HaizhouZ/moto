#ifndef MOTO_SOLVER_LCID_RICCATI_LCID_SOLVER_HPP
#define MOTO_SOLVER_LCID_RICCATI_LCID_SOLVER_HPP

#include <moto/multibody/lcid.hpp>
#include <moto/multibody/stacked_euler.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/spmm/sparse_mat.hpp>

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
    ns_riccati_data create_data(node_data *full_data) override;

    custom_func lcid_;
    constr euler_;
    using func_type = multibody::lcid;
    using int_type = multibody::stacked_euler;

    void ns_factorization(ns_riccati_data *cur) override;
    void riccati_recursion(ns_riccati_data *cur, ns_riccati_data *prev) override;
    void riccati_recursion_correction(ns_riccati_data *cur, ns_riccati_data *prev) override;
    void compute_primal_sensitivity(ns_riccati_data *cur) override;
    void compute_primal_sensitivity_correction(ns_riccati_data *cur) override;
    void finalize_primal_step(ns_riccati_data *cur, bool finalize_dual) override;
    void finalize_primal_step_correction(ns_riccati_data *cur) override;
    void finalize_dual_newton_step(ns_riccati_data *cur);
    struct MOTO_ALIGN_NO_SHARING data : public ns_riccati_data::aux_data {
        sparse_mat sp_u_y_K; // [a_y; f_y; tq_y]_K
        size_t sp_u_y_K_a_f_st, sp_u_y_K_a_f_ed;
        aligned_map_t sp_u_y_K_a_f;
        sparse_mat sp_Z_u; // [Z_a; Z_f; Z_tq]
        aligned_map_t sp_Z_u_a, sp_Z_u_f;
        sparse_mat &f_x, &f_u; // ref to unprojected dynamics derivatives
        matrix Q_yy_p;         // projected Q_yy
        vector z_step;         // nullspace step

        sparse_mat Q_yy_F_u;
        aligned_map_t Q_yy_F_u_;
        matrix Q_yy_F_u_Z_u;
        sparse_mat F_u_T_Q_yy_F_u_Z_u;
        aligned_map_t F_u_T_Q_yy_F_u_Z_u_;
        sparse_mat F_u_T_buf_data;
        aligned_map_t F_u_T_buf_data_;
        data(ns_riccati_data &d, lcid_solver &solver);
    };

    lcid_solver(const func_type &l, const int_type &e) : lcid_(l), euler_(e) {}
};
} // namespace lcid_riccati

} // namespace solver
} // namespace moto

#endif