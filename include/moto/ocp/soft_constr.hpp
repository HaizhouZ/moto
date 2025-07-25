#ifndef MOTO_OCP_SOFT_CONSTR_HPP
#define MOTO_OCP_SOFT_CONSTR_HPP

#include <moto/ocp/constr.hpp>

namespace moto {
/**
 * @brief soft constraint interface class
 * @warning jacobian modification should be added to @ approx_map::jac_modification_
 */
class soft_constr : public constr {
  private:
    using base = constr;

  public:
    /**
     * @brief soft constraint data, contains:
     * 1. additional primal step data for splitting post-rollout operation
     * 2. jacobian modification data for the soft constraint
     *
     */
    struct approx_map : public base::approx_map {
        std::vector<vector_ref> prim_step_;            // to be set
        std::vector<row_vector_ref> jac_modification_; ///< merit jacobian modification
        using map_base = base::approx_map;
        approx_map(dense_approx_data &raw, map_base &&rhs) : map_base(std::move(rhs)) {
            map_merit_jac_from_raw(raw.jac_modification_, jac_modification_);
        }
    };

    /// public type alias for @ref approx_map to ensure common interface of all soft constraints
    using data_map_t = approx_map;

    struct impl : public base::impl {
        using base::impl::impl;                                ///< inherit constructors
        impl(base::impl &&rhs) : base::impl(std::move(rhs)) {} ///< move constructor from base impl

        /// @brief finalize the soft constraint, will be called upon added to a problem
        void finalize_impl() override;

        bool skip_field_check = false; ///< skip field check in finalize_impl

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
    };

    DEF_IMPL_GETTER();

  public:
    using base::base; ///< inherit constructors

    // soft_constr(base &&rhs) : base(std::move(rhs)) { field_hint().is_soft = true; } ///< move constructor from constr
    /***
     * @brief make approximation data for the soft constraint, will use default @ref data_type
     */
    func_approx_map_ptr_t create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) const override {
        return func_approx_map_ptr_t(make_approx<soft_constr>(primal, raw, shared));
    }
};
} // namespace moto

#endif // MOTO_OCP_SOFT_CONSTR_HPP