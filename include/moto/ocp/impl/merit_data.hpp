#ifndef MOTO_OCP_PROBLEM_DATA_HPP
#define MOTO_OCP_PROBLEM_DATA_HPP

#include <moto/core/array.hpp>
#include <moto/core/fields.hpp>
#include <moto/spmm/sparse_mat.hpp>
namespace moto {
class ocp;
struct generic_dynamics;
/**
 * @brief dense raw approximation data
 * deserialized data storage of all function fields
 */
struct merit_data {
    merit_data(ocp *prob);

    ocp *prob_;
    struct approx_data {
        vector v_; // value
        /// outer index is field, inner index is dynamics index
        array_type<sparse_mat, primal_fields> jac_;
    };
    array_type<approx_data, constr_fields> approx_;
    constexpr static auto stored_constr_fields = constr_fields;
    struct dynamics_data {
        sparse_mat proj_f_x_, proj_f_u_;
        vector proj_f_res_;
    };
    auto &proj_f_x() { return dynamics_data_.proj_f_x_; }
    auto &proj_f_u() { return dynamics_data_.proj_f_u_; }
    auto &proj_f_res() { return dynamics_data_.proj_f_res_; }
    dynamics_data dynamics_data_;
    /// dual variables of constratins, indexed by field
    array_type<vector, constr_fields> dual_;
    /// complementarity of each inequality fields
    array_type<vector, ineq_constr_fields> comp_;
    scalar_t merit_; ///< cost + sum of all constraints multipler-residual products
    scalar_t cost_;  ///< cost value
    /// cost jacobian (pure cost gradient; excludes constraint dual contributions J_c^T λ)
    array<row_vector, field::num_prim> cost_jac_;
    /// Lagrangian jacobian: cost_jac_ + Σ_c J_c^T λ_c (used for dual residual / stationarity)
    array<row_vector, field::num_prim> merit_jac_;
    /// modification of the merit jacobian, indexed by field
    array<row_vector, field::num_prim> merit_jac_modification_;
    /// cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    array<array<sparse_mat, field::num_prim>, field::num_prim> hessian_;
    array<array<sparse_mat, field::num_prim>, field::num_prim> hessian_modification_;

    /// stationary residual
    array_type<row_vector, primal_fields> res_stat_;

    array_type<aligned_vector_map_t, primal_fields> primal_prox_hess_diagonal_;
};
def_unique_ptr(merit_data);
} // namespace moto

#endif // MOTO_OCP_PROBLEM_DATA_HPP