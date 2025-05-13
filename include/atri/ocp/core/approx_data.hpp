#ifndef ATRI_OCP_PROBLEM_DATA_HPP
#define ATRI_OCP_PROBLEM_DATA_HPP

#include <atri/core/offset_array.hpp>
#include <atri/ocp/core/problem.hpp>

namespace atri {

/**
 * @brief dense raw approximation data
 * deserialized data storage of all function fields
 */
struct approx_data {
    approx_data(problem_ptr_t prob);

    problem_ptr_t prob_;
    struct raw_approx {
        vector v_;                                // value
        std::array<matrix, field::num_prim> jac_; // jacobian
    };
    offset_array<raw_approx, field::num_constr, __dyn> approx_;
    offset_array<vector, field::num_constr, __dyn> dual_;
    vector cost_;
    // cost jacobian
    row_vector jac_[field::num_prim];
    // cost hessian h[a][b] is h_ab. Note only the upper block-triangular part is stored
    std::array<std::array<matrix, field::num_prim>, field::num_prim> hessian_;
};

} // namespace atri

#endif // ATRI_OCP_PROBLEM_DATA_HPP