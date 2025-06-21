#ifndef ATRI_OCP_PROBLEM_DATA_HPP
#define ATRI_OCP_PROBLEM_DATA_HPP

#include <atri/core/array.hpp>
#include <atri/ocp/problem.hpp>

namespace atri {

/**
 * @brief dense raw approximation data
 * deserialized data storage of all function fields
 */
struct approx_storage {
    approx_storage(const problem_ptr_t &prob);

    problem_ptr_t prob_;
    struct raw_approx {
        vector v_;                           // value
        array<matrix, field::num_prim> jac_; // jacobian
    };
    shifted_array<raw_approx, field::num_constr, __dyn> approx_;
    shifted_array<vector, field::num_constr, __dyn> dual_;
    scalar_t cost_;
    // cost jacobian
    array<row_vector, field::num_prim> jac_;
    // cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    array<array<matrix, field::num_prim>, field::num_prim> hessian_;
};
def_unique_ptr(approx_storage);
} // namespace atri

#endif // ATRI_OCP_PROBLEM_DATA_HPP