#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_config.hpp>

namespace moto {
namespace solver {

class ipm_constr final : public ineq_constr {
  private:
    using base = ineq_constr;

  public:
    struct approx_data : public base::approx_data {
        ipm_config *ipm_cfg = nullptr; ///< pointer to the IPM settings
        vector g_;                     ///< ipm primal value
        vector r_s_;                   ///< ipm residuals g + t
        vector slack_;                 ///< slack variables for the constraints
        vector diag_scaling;           ///< Nesterov-Todd scaling T^{-1} N
        vector scaled_res_;            ///< residuals after NT scaling (Nr_g - r_s) T^{-1} = T{-1} N r_g + T^{-1} mu
        vector d_slack_;               ///< newton step for slack variables
        vector corrector_;             ///< newton step for multipliers
        vector reg_;
        vector active_;
        vector reg_T_inv_;
        approx_data(base::approx_data &&rhs);
    };
    using base::base;
    using ipm_data = data_type<ipm_constr>;
    /// update the IPM slack and residuals
    void value_impl(func_approx_data &data) const override final;
    /// update the IPM-modified cost jacobian and hessian
    void jacobian_impl(func_approx_data &data) const override final;

    void propagate_jacobian(func_approx_data &d) const override;
    void propagate_hessian(func_approx_data &d) const override;
    void propagate_res_stats(func_approx_data &d) const {};

  public:
    void setup_workspace_data(func_arg_map &data, workspace_data *settings) const override {
        base::setup_workspace_data(data, settings);
        data.as<ipm_data>().ipm_cfg = &settings->as<ipm_config>();
    }
    /// @brief initialize the IPM constraint data
    void initialize(data_map_t &data) const override final;
    /// @brief post rollout operation for the IPM constraint to compute the newton step
    void finalize_newton_step(data_map_t &data) const override final;
    /// @brief finalize the predictor step, should be called after the rollout
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) const override final;
    /// @brief will compute the cost jacobian correction depending on the IPM settings
    void apply_corrector_step(data_map_t &data) const override final;
    /// @brief line search step for the IPM constraint
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override final;
    /// @brief update the line search configuration (if necessary)
    void update_linesearch_bounds(data_map_t &data, workspace_data *cfg) const override final;
    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return func_approx_data_ptr_t
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<ipm_constr>(primal, raw, shared));
    }
    DEF_FUNC_CLONE;
};
} // namespace solver
using ipm = solver::ipm_constr;

} // namespace moto

#include <moto/ocp/constr.hpp>

#endif // MOTO_SOLVER_IPM_HPP