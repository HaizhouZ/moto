#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/impl/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_config.hpp>

namespace moto {
namespace solver {

class ipm_constr final : public impl::ineq_constr {
  private:
    using base = impl::ineq_constr;

  public:
    struct approx_data : public base::approx_data {
        ipm_config *ipm_cfg = nullptr; ///< pointer to the IPM settings
        vector g_;                     ///< ipm primal value
        vector r_s_;                   ///< ipm residuals g + t
        vector slack_;                 ///< slack variables for the constraints
        vector diag_scaling;           ///< Nesterov-Todd scaling T^{-1} N
        vector scaled_res_;            ///< residuals after NT scaling (Nr_g - r_s) T^{-1} = T{-1} N r_g + T^{-1} mu
        vector d_slack_;               ///< newton step for slack variables
        vector d_multipler_;           ///< newton step for multipliers
        template <typename approx_data_t>
        approx_data(approx_data_t &&rhs)
            : approx_data_t(std::move(rhs)) {
            slack_.resize(f_->dim_);
            diag_scaling.resize(f_->dim_);
            scaled_res_.resize(f_->dim_);
        }
    };

  private:
    /// + update the IPM slack and residuals
    void value_impl(func_approx_map &data) override final;
    /// + update the IPM-modified cost jacobian and hessian
    void jacobian_impl(func_approx_map &data) override final;

  public:
    using base::base;
    void setup_workspace_data(func_arg_map &data, workspace_data *settings) override {
        base::setup_workspace_data(data, settings);
        data.as<approx_data>().ipm_cfg = &settings->get<ipm_config>();
    }
    /// @brief initialize the IPM constraint data
    void initialize(data_map_t &data) override final;
    /// @brief post rollout operation for the IPM constraint to compute the newton step
    void finalize_newton_step(data_map_t &data) override final;
    /// @brief finalize the predictor step, should be called after the rollout
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) override final;
    /// @brief will compute the cost jacobian correction depending on the IPM settings
    void correct_jacobian(data_map_t &data) override final;
    /// @brief line search step for the IPM constraint
    void line_search_step(data_map_t &data, workspace_data *cfg) override final;
    /// @brief update the line search configuration (if necessary)
    void update_linesearch_config(data_map_t &data, workspace_data *cfg) override final;

    using ipm_data = data_type<ipm_constr>;

    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return func_approx_map_ptr_t
     */
    func_approx_map_ptr_t create_approx_map(sym_data &primal, dense_approx_data &raw, shared_data &shared) override {
        return func_approx_map_ptr_t(make_approx<ipm_constr>(primal, raw, shared));
    }

  private:
    void propagate_jacobian(ipm_data &d);
    void propagate_hessian(ipm_data &d);
};
} // namespace solver
using ipm = solver::ipm_constr;

} // namespace moto

#include <moto/ocp/constr.hpp>

#endif // MOTO_SOLVER_IPM_HPP