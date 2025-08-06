#ifndef MOTO_OCP_PROBLEM_DATA_HPP
#define MOTO_OCP_PROBLEM_DATA_HPP

#include <moto/core/array.hpp>
#include <moto/core/fields.hpp>

namespace moto {
class ocp;
/**
 * @brief dense raw approximation data
 * deserialized data storage of all function fields
 */
struct merit_data {
    merit_data(ocp *prob);

    ocp *prob_;
    struct raw_approx {
        vector v_;                              ///< dense value
        array<matrix_rm, field::num_prim> jac_; ///< dense jacobian
    };
    /// field approximation stored in here
    static constexpr auto stored_constr_fields = std::array{__dyn, __eq_x, __eq_xu};
    /// raw approximation data of constraints, indexed by field
    array_type<raw_approx, stored_constr_fields> approx_;
    /// dual variables of constratins, indexed by field
    array_type<vector, constr_fields> dual_;
    /// complementarity of each inequality fields
    array_type<vector, ineq_constr_fields> comp_;
    scalar_t merit_; ///< cost + sum of all constraints multipler-residual products
    scalar_t cost_;  ///< cost value
    /// cost jacobian
    array<row_vector, field::num_prim> jac_;
    /// modification of the merit jacobian, indexed by field
    array<row_vector, field::num_prim> jac_modification_;
    /// cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    array<array<matrix, field::num_prim>, field::num_prim> hessian_;

    struct active_ineq {
        scalar_t *d_multiplier_; ///< pointer to the multiplier of the active inequality constraint
    };
    array_type<std::vector<active_ineq>, ineq_constr_fields> active_ineqs_; ///< active inequality constraints
    struct raw_ineq_approx : public raw_approx {
        size_t dim = 0;                                               ///< dimension of the active inequality constraint
        auto v() { return v_.head(dim); } ///< dense value of the active inequality constraint
        auto jac(field_t f) { return jac_[f].topRows(dim); } ///< jacobian of the active inequality constraint
    };
    array_type<raw_ineq_approx, ineq_constr_fields> active_ineq_approx_;

    /// stationary residual
    array_type<row_vector, primal_fields> res_stat_;
};
def_unique_ptr(merit_data);
} // namespace moto

#endif // MOTO_OCP_PROBLEM_DATA_HPP