#ifndef MOTO_SOLVER_DATA_BASE_HPP
#define MOTO_SOLVER_DATA_BASE_HPP

#include <moto/ocp/node_data.hpp>

namespace moto {
namespace solver {
constexpr field_t primal_fields[] = {__x, __u, __y};

/**
 * @brief default solver data class, stores some shortcuts for solver implementation,
 * and also an array of primal (newton) step for later linear rollout
 * @note this class can be used as base class for other solver data (optional)
 */
struct data_base : public node_data {
    size_t nx, nu;
    // value function
    row_vector &Q_x;
    row_vector &Q_u;
    row_vector &Q_y;
    matrix &Q_xx;
    matrix &Q_ux;
    matrix &Q_uu;
    matrix &Q_yx;
    matrix &Q_yy;
    array<vector, std::size(primal_fields)> prim_step; ///< primal (newton) step
    /// @brief create solver data
    /// @param prob ocp to initialize nx, nu
    /// @param dense_ to initialize Q_ ref and rollout_ primal data
    data_base(const ocp_ptr_t &prob);
};

} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_DATA_BASE_HPP