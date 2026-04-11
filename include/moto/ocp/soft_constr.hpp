#ifndef MOTO_OCP_SOFT_CONSTR_HPP
#define MOTO_OCP_SOFT_CONSTR_HPP

#include <moto/ocp/constr.hpp>

namespace moto {
namespace solver {
struct data_base;
}
/**
 * @brief soft constraint interface class
 * @warning lagrangian gradient correction should be added to @ approx_data::lag_jac_corr_
 */
class soft_constr : public generic_constr {
  private:
    using base = generic_constr;

  public:
    /**
     * @brief soft constraint data, contains:
     * 1. additional primal step data for splitting post-rollout operation
     * 2. lagrangian gradient correction data for the soft constraint
     *
     */
    struct approx_data : public base::approx_data {
        std::vector<vector_ref> prim_step_;            // to be set
        std::vector<row_vector_ref> lag_jac_corr_; ///< lagrangian gradient correction
        mapped_vector d_multiplier_;                          ///< newton step for multipliers
        const vector *jacobian_row_scaling_ = nullptr;
        const vector *hessian_diag_scaling_ = nullptr;
        workspace_data *ws_data_ = nullptr;
        solver::data_base *solver_data_ = nullptr;
        bool runtime_bound_ = false;
        bool initialized_ = false;

        using data_base = base::approx_data;
        approx_data(data_base &&rhs);

    };

    /// public type alias for @ref approx_data to ensure common interface of all soft constraints
    using data_map_t = approx_data;

  protected:
    bool skip_field_check = false; ///< skip field check in finalize_impl

    /// @brief finalize the soft constraint, will be called upon added to a problem
    void finalize_impl() override;

    public:
    using base::base; ///< inherit constructors
    soft_constr(generic_constr &&rhs) : base(std::move(rhs)) {
        field_hint().is_soft = true; ///< set the field hint to soft
    } ///< move constructor from generic_constr
    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override {
        base::setup_workspace_data(data, ws_data);
        auto &sd = data.as<data_map_t>();
        sd.ws_data_ = ws_data;
    }
    /// initialize the soft constraint data
    virtual void initialize(data_map_t &data) const = 0;
    /// post rollout operation for the soft constraint to compute the newton step
    virtual void finalize_newton_step(data_map_t &data) const = 0;
    /// @brief finalize the predictor step, should be called after the rollout
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be finalized
    virtual void finalize_predictor_step(data_map_t &data, workspace_data *worker_cfg) const {};
    /// first order correction of the Lagrangian gradient. lag_jac_corr_ must be reset to zero before calling this
    virtual void apply_corrector_step(data_map_t &data) const {};
    /// @brief line search step for the soft constraint
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be used
    virtual void apply_affine_step(data_map_t &data, workspace_data *worker_cfg) const = 0;
    /// @brief update the line search configuration (if necessary)
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be updated
    virtual void update_ls_bounds(data_map_t &data, workspace_data *worker_cfg) const {}
    /// @brief backup the current line-search trial state
    /// @param data data map
    virtual void backup_trial_state(data_map_t &data) const {}
    /// @brief restore the current line-search trial state
    /// @param data data map
    virtual void restore_trial_state(data_map_t &data) const {}

    /// @brief contribution to the phase original objective (for example exact penalties)
    virtual scalar_t objective_penalty(const func_approx_data &data) const { return 0.; }
    /// @brief directional derivative of @ref objective_penalty along the current trial step
    virtual scalar_t objective_penalty_dir_deriv(const func_approx_data &data) const { return 0.; }
    /// @brief contribution to the phase search barrier / penalization objective
    virtual scalar_t search_penalty(const func_approx_data &data) const { return 0.; }
    /// @brief directional derivative of @ref search_penalty along the current trial step
    virtual scalar_t search_penalty_dir_deriv(const func_approx_data &data) const { return 0.; }
    /// @brief optional local stationarity residual summary owned by the constraint
    virtual scalar_t local_stat_residual_inf(const func_approx_data &data) const { return 0.; }
    /// @brief optional local complementarity summary owned by the constraint
    virtual scalar_t local_comp_residual_inf(const func_approx_data &data) const { return 0.; }

    // soft_constr(base &&rhs) : base(std::move(rhs)) { field_hint().is_soft = true; } ///< move constructor from generic_constr
    /***
     * @brief make approximation data for the soft constraint, will use default @ref data_type
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<soft_constr>(primal, raw, shared));
    }
};
} // namespace moto

#endif // MOTO_OCP_SOFT_CONSTR_HPP
