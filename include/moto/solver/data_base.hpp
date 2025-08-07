#ifndef MOTO_SOLVER_DATA_BASE_HPP
#define MOTO_SOLVER_DATA_BASE_HPP

#include <moto/ocp/impl/merit_data.hpp>

namespace moto {
class sym_data;
namespace solver {

/**
 * @brief default solver data class, stores some shortcuts for solver implementation,
 * and also an array of primal (newton) step for later linear rollout
 * @note this class can be used as base class for other solver data (optional)
 */
struct data_base {
    size_t nx, nu, ny; ///< dimensions of the problem
    sym_data *sym_;
    merit_data *dense_; ///< pointer to the dense approximation data
    // value function
    row_vector &Q_x, Q_x_bak;
    row_vector &Q_u, Q_u_bak;
    row_vector &Q_y, Q_y_bak;
    matrix &Q_xx, Q_xx_bak;
    matrix &Q_ux, Q_ux_bak;
    matrix &Q_uu, Q_uu_bak;
    matrix &Q_yx, Q_yx_bak;
    matrix &Q_yy, Q_yy_bak;
    array_type<vector, primal_fields> prim_step; ///< primal (newton) step
    array_type<vector, primal_fields> prim_corr; ///< correction for the primal step
    row_vector *Q_y_corr;                        ///< correction for the Q_y

    array_type<vector, constr_fields> dual_step; // dual rollout

    /// @brief create solver data
    /// @param sym_ pointer to the symbolic data
    /// @param dense pointer to the dense approximation data
    data_base(sym_data *sym_, merit_data *dense);
    data_base(const data_base &rhs) = delete;
    data_base(data_base &&rhs) = default;
    void merge_jacobian_modification();
    void swap_jacobian_modification();
    virtual ~data_base() = default;
};
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_DATA_BASE_HPP