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
struct MOTO_ALIGN_NO_SHARING data_base {
    size_t nx, nu, ny; ///< dimensions of the problem
    sym_data *sym_;
    merit_data *dense_; ///< pointer to the dense approximation data
    // value function
    row_vector &Q_x, Q_x_bak;
    row_vector &Q_u, Q_u_bak;
    row_vector &Q_y, Q_y_bak;
    sparse_mat &Q_xx, &Q_xx_mod;
    sparse_mat &Q_ux, &Q_ux_mod;
    sparse_mat &Q_uu, &Q_uu_mod;
    sparse_mat &Q_yx, &Q_yx_mod;
    sparse_mat &Q_yy, &Q_yy_mod;
    matrix V_xx, V_yy;
    array_type<vector, primal_fields> trial_prim_step;      ///< primal (newton) trial step
    array_type<vector, primal_fields> prim_corr;            ///< correction for the primal step
    array_type<vector, primal_fields> trial_prim_state_bak; ///< backup of the original primal state for line search

    array_type<vector, constr_fields> trial_dual_step;      // dual rollout trial step
    array_type<vector, constr_fields> trial_dual_state_bak; ///< backup of the original dual state for line search

    /// @brief create solver data
    /// @param sym_ pointer to the symbolic data
    /// @param dense pointer to the dense approximation data
    data_base(sym_data *sym_, merit_data *dense);
    data_base(const data_base &rhs) = delete;
    data_base(data_base &&rhs) = default;
    void merge_jacobian_modification();
    void swap_jacobian_modification();
    void backup_trial_state();
    void restore_trial_state();
    /// @brief first-order correction step, will clear the jacobian modification and backup the primal trial state, then call the callback to fill in the jacobian modification, finally swap the modification into the jacobian for later use
    /// @param callback Callback called after clearing modification, so it should fill in the jacobian modification, can be a lambda or std::function, and should be invocable with either data_base* or void
    template <typename Callback>
    void first_order_correction_start(Callback &&callback) {
        prim_corr[__x].setZero();
        // clear modification
        for (auto field : primal_fields) {
            dense_->jac_modification_[field].setZero();
        }
        if constexpr (std::is_invocable_v<Callback, data_base *>) {
            callback(this);
        } else if constexpr (std::is_invocable_v<Callback>) {
            callback();
        } else {
            static_assert(false, "Callback must be invocable with data_base* or void");
        }
        swap_jacobian_modification(); // move modification to the jacobian for later solving
    }
    void first_order_correction_end() {
        swap_jacobian_modification();
    }
    virtual ~data_base() = default;
};
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_DATA_BASE_HPP