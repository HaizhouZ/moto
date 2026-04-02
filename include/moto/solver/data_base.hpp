#ifndef MOTO_SOLVER_DATA_BASE_HPP
#define MOTO_SOLVER_DATA_BASE_HPP

#include <moto/ocp/impl/lag_data.hpp>

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
    lag_data *dense_; ///< pointer to the dense approximation data
    // Active stage gradient used by the current linear solve.
    // These alias dense_->lag_jac_[...], which stores the base stage Lagrangian
    // gradient plus any currently-activated pending correction.
    //
    // Q_y is still just the stage gradient with respect to next-state coordinates.
    // During backward recursion it is also the local input to cross-stage costate
    // propagation, but it is not itself a separate value-function object.
    row_vector &Q_x;
    row_vector &Q_u;
    row_vector &Q_y;
    sparse_mat &Q_xx, &Q_xx_mod;
    sparse_mat &Q_ux, &Q_ux_mod;
    sparse_mat &Q_uu, &Q_uu_mod;
    sparse_mat &Q_yx, &Q_yx_mod;
    sparse_mat &Q_yy, &Q_yy_mod;
    // Snapshot of the base stage Lagrangian gradient before any pending
    // reduced-system correction in lag_jac_corr_ is activated.
    //
    // In restoration mode this is the original stagewise restoration
    // Lagrangian gradient on w=(x,u,y); condensed elastic terms remain in
    // lag_jac_corr_ and are not part of this backup.
    array_type<row_vector, primal_fields> base_lag_grad_backup;
    array_type<row_vector, primal_fields> kkt_stat_err_;
    matrix V_xx, V_yy;
    array_type<vector, primal_fields> trial_prim_step;      ///< primal (newton) trial step
    array_type<vector, primal_fields> prim_corr;            ///< correction for the primal step
    array_type<vector, primal_fields> trial_prim_state_bak; ///< backup of the original primal state for line search

    array_type<vector, constr_fields> trial_dual_step;      // dual rollout trial step
    array_type<vector, constr_fields> trial_dual_state_bak; ///< backup of the original dual state for line search

    /// @brief create solver data
    /// @param sym_ pointer to the symbolic data
    /// @param dense pointer to the dense approximation data
    data_base(sym_data *sym_, lag_data *dense);
    data_base(const data_base &rhs) = delete;
    data_base(data_base &&rhs) = default;
    void activate_lag_jac_corr();
    void swap_active_and_lag_jac_corr();
    void backup_trial_state();
    void restore_trial_state();
    /// @brief Prepare a first-order correction solve.
    /// @details The callback writes the correction into dense_->lag_jac_corr_.
    /// Then that buffer is swapped with dense_->lag_jac_ so the correction system
    /// can be solved without materializing an extra gradient array.
    template <typename Callback>
    void first_order_correction_start(Callback &&callback) {
        prim_corr[__x].setZero();
        for (auto field : primal_fields) {
            dense_->lag_jac_corr_[field].setZero();
        }
        if constexpr (std::is_invocable_v<Callback, data_base *>) {
            callback(this);
        } else if constexpr (std::is_invocable_v<Callback>) {
            callback();
        } else {
            static_assert(false, "Callback must be invocable with data_base* or void");
        }
        swap_active_and_lag_jac_corr();
    }
    void first_order_correction_end() {
        swap_active_and_lag_jac_corr();
    }
    virtual ~data_base() = default;
};
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_DATA_BASE_HPP
