#ifndef MOTO_UTILS_FIELD_CONVERSION_HPP
#define MOTO_UTILS_FIELD_CONVERSION_HPP

#include <moto/core/expr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
class ocp;
namespace utils {
/**
 * @brief backward copy from next stacked y to current x, usually to ensure consistency of initialization
 */
void copy_y_to_x(vector_ref from_y, vector_ref to_x, const ocp *prob_y, const ocp *prob_x);
void copy_y_to_x_tangent(vector_ref from_y, vector_ref to_x, const ocp *prob_y, const ocp *prob_x);
/**
 * @brief forward copy from stacked x to y
 */
void copy_x_to_y(vector_ref from_x, vector_ref to_y, const ocp *prob_x, const ocp *prob_y);
void copy_x_to_y_tangent(vector_ref from_x, vector_ref to_y, const ocp *prob_x, const ocp *prob_y);

/**
 * @brief Get the permutation matrix that maps y to x, i.e., X = P * Y.
 * for example: derivative conversion: dfdy = dfdx * P
 * @param prob_y problem of which the y field (order of syms) is used
 * @param prob_x problem of which the x field (order of syms) is used
 * @return Eigen::PermutationMatrix<-1, -1>& the permutation matrix that maps y to x
 */
Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp *prob_y, const ocp *prob_x);
} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_FIELD_CONVERSION_HPP