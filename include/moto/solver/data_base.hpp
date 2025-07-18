#ifndef MOTO_SOLVER_DATA_BASE_HPP
#define MOTO_SOLVER_DATA_BASE_HPP

#include <moto/ocp/impl/node_data.hpp>

namespace moto {
namespace solver {

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
    array_type<vector, primal_fields> prim_step; ///< primal (newton) step
    array_type<vector, primal_fields> prim_corr; ///< correction for the primal step
    row_vector Q_y_cache; ///< cache before correction for the Q_y
    /// @brief create solver data
    /// @param prob ocp to initialize nx, nu and the Q-derivative refs
    data_base(const ocp_ptr_t &prob);
};
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_DATA_BASE_HPP