#ifndef MOTO_OCP_SOFT_CONSTR_HPP
#define MOTO_OCP_SOFT_CONSTR_HPP

#include <moto/ocp/constr.hpp>

namespace moto {
/**
 * @brief soft constraint interface class
 * @warning jacobian modification should be added to @ approx_data::jac_modification_
 */
class soft_constr : public generic_constr {
  private:
    using base = generic_constr;

  public:
    /**
     * @brief soft constraint data, contains:
     * 1. additional primal step data for splitting post-rollout operation
     * 2. jacobian modification data for the soft constraint
     *
     */
    struct approx_data : public base::approx_data {
        std::vector<vector_ref> prim_step_;            // to be set
        std::vector<row_vector_ref> jac_modification_; ///< merit jacobian modification
        mapped_vector d_multiplier_;                          ///< newton step for multipliers

        using data_base = base::approx_data;
        approx_data(data_base &&rhs) : data_base(std::move(rhs)), d_multiplier_(nullptr, 0) {
            map_merit_jac_from_raw(merit_data_->jac_modification_, jac_modification_);
            // d_multiplier_.resize(func_.dim());
            // d_multiplier_.setZero();
        }
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
    /// initialize the soft constraint data
    virtual void initialize(data_map_t &data) const = 0;
    /// post rollout operation for the soft constraint to compute the newton step
    virtual void finalize_newton_step(data_map_t &data) const = 0;
    /// @brief finalize the predictor step, should be called after the rollout
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be finalized
    virtual void finalize_predictor_step(data_map_t &data, workspace_data *worker_cfg) const {};
    /// first order correction of the cost jacobian. jac_modification must be reset to zero before calling this
    virtual void correct_jacobian(data_map_t &data) const {};
    /// @brief line search step for the soft constraint
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be used
    virtual void line_search_step(data_map_t &data, workspace_data *worker_cfg) const = 0;
    /// @brief update the line search configuration (if necessary)
    /// @param data data map
    /// @param worker_cfg workspace data pointer to the config to be updated
    virtual void update_linesearch_config(data_map_t &data, workspace_data *worker_cfg) const {}
    // soft_constr(base &&rhs) : base(std::move(rhs)) { field_hint().is_soft = true; } ///< move constructor from generic_constr
    /***
     * @brief make approximation data for the soft constraint, will use default @ref data_type
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<soft_constr>(primal, raw, shared));
    }
};
} // namespace moto

#endif // MOTO_OCP_SOFT_CONSTR_HPP