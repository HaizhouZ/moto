#ifndef MOTO_OCP_PROBLEM_DATA_HPP
#define MOTO_OCP_PROBLEM_DATA_HPP

#include <moto/core/array.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
/**
 * @brief dense raw approximation data
 * deserialized data storage of all function fields
 */
struct approx_storage {
    approx_storage(const ocp_ptr_t &prob);

    ocp_ptr_t prob_;
    struct raw_approx {
        vector v_;                           ///< dense value
        array<matrix, field::num_prim> jac_; ///< dense jacobian
    };
    /// raw approximation data of constraints, indexed by field
    shifted_array<raw_approx, field::num_constr, __dyn> approx_;
    /// dual variables of constratins, indexed by field
    shifted_array<vector, field::num_constr, __dyn> dual_;
    scalar_t merit_;
    scalar_t cost_;
    /// cost jacobian
    array<row_vector, field::num_prim> jac_;
    /// cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    array<array<matrix, field::num_prim>, field::num_prim> hessian_;
};
def_unique_ptr(approx_storage);
} // namespace moto

#endif // MOTO_OCP_PROBLEM_DATA_HPP