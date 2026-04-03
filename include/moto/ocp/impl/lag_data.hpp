#ifndef MOTO_OCP_LAG_DATA
#define MOTO_OCP_LAG_DATA

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
struct lag_data {
    lag_data(ocp *prob);

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
    scalar_t lag_; ///< cost + sum of all constraints multipler-residual products
    scalar_t cost_;  ///< cost value
    /// cost jacobian (pure cost gradient; excludes constraint dual contributions J_c^T λ)
    array<row_vector, field::num_prim> cost_jac_;
    /// Base stage Lagrangian gradient:
    ///   cost_jac_ + Σ_c J_c^T λ_c
    /// This is the persistent gradient state produced by update_approximation().
    /// Solver code often aliases these entries as Q_x / Q_u / Q_y.
    array<row_vector, field::num_prim> lag_jac_;
    /// Pending additive correction to lag_jac_ used by the next linear solve.
    /// Typical writers are:
    /// - IPM / PMM Schur-complement terms
    /// - iterative refinement residual correction
    ///
    /// Lifecycle:
    /// - update_approximation() clears this buffer
    /// - solver code fills it
    /// - data_base::activate_lag_jac_corr() adds it into lag_jac_
    /// - some correction paths temporarily swap this buffer with lag_jac_ for efficiency
    array<row_vector, field::num_prim> lag_jac_corr_;
    /// cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    array<array<sparse_mat, field::num_prim>, field::num_prim> lag_hess_;
    array<array<sparse_mat, field::num_prim>, field::num_prim> hessian_modification_;
};
def_unique_ptr(lag_data);
} // namespace moto

#endif // MOTO_OCP_LAG_DATA
