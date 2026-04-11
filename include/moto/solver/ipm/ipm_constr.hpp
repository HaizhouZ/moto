#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/ipm/positivity_step.hpp>

namespace moto {
namespace solver {

class ipm_constr : public ineq_constr {
  private:
    using base = ineq_constr;

  public:
    /**
     * @brief Approximation data for the IPM constraints
     * @note data.v_ will store g + t (t is the slack variable)
     */
    struct approx_data : public base::approx_data {
        struct side_data : public base::approx_data::box_pair_runtime {
            vector r_s;
            vector reg;
            vector reg_T_inv;
            vector diag_scaling;
            vector scaled_res;
            vector corrector;

            void resize(Eigen::Index n) override {
                base::approx_data::box_pair_runtime::resize(n);
                r_s.setZero(n);
                reg.setZero(n);
                reg_T_inv.setZero(n);
                diag_scaling.setZero(n);
                scaled_res.setZero(n);
                corrector.setZero(n);
            }
        };

        ipm_config *ipm_cfg = nullptr; ///< pointer to the IPM settings
        vector lifted_view;
        vector jac_step;
        approx_data(base::approx_data &&rhs);
    };
    using base::base;
    ipm_constr(const std::string &name, approx_order order, size_t dim, field_t field = field_t::__undefined)
        : base(name, order, dim, field) {
        synthesize_upper_half_box_info_if_missing();
    }
    ipm_constr(const std::string &name,
               const var_inarg_list &args,
               const cs::SX &out,
               approx_order order,
               field_t field = field_t::__undefined)
        : base(name, args, out, order, field) {
        synthesize_upper_half_box_info_if_missing();
    }
    ipm_constr(generic_constr &&rhs) : base(std::move(rhs)) {
        synthesize_upper_half_box_info_if_missing();
    }
    using ipm_data = data_type<ipm_constr>;
    /// update the IPM slack and residuals
    void value_impl(func_approx_data &data) const override;
    /// update the IPM-modified cost jacobian and hessian
    void jacobian_impl(func_approx_data &data) const override;
    void propagate_jacobian(func_approx_data &data) const;
    void propagate_hessian(func_approx_data &data) const;
    std::unique_ptr<base::approx_data::box_pair_runtime> create_side_data() const override {
        return std::make_unique<approx_data::side_data>();
    }
    residual_summary primal_residual_summary(const func_approx_data &data) const override;

  public:
    void setup_workspace_data(func_arg_map &data, workspace_data *settings) const override {
        base::setup_workspace_data(data, settings);
        data.as<ipm_data>().ipm_cfg = &settings->as<ipm_config>();
    }
    /// @brief initialize the IPM constraint data
    void initialize(data_map_t &data) const override;
    /// @brief post rollout operation for the IPM constraint to compute the newton step
    void finalize_newton_step(data_map_t &data) const override;
    /// @brief finalize the predictor step, should be called after the rollout
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) const override;
    /// @brief will compute the cost jacobian correction depending on the IPM settings
    void apply_corrector_step(data_map_t &data) const override;
    /// @brief line search step for the IPM constraint
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override;
    /// @brief update the line search configuration (if necessary)
    void update_ls_bounds(data_map_t &data, workspace_data *cfg) const override;
    /// @brief backup the current IPM trial state before a line-search attempt
    void backup_trial_state(data_map_t &data) const override;
    /// @brief restore the backed-up IPM trial state before the next line-search attempt
    void restore_trial_state(data_map_t &data) const override;
    void restoration_commit_dual_step(data_map_t &data, scalar_t alpha_dual) const override;
    void restoration_reset_bound_multipliers(data_map_t &data) const override;
    scalar_t search_penalty(const func_approx_data &data) const override;
    scalar_t search_penalty_dir_deriv(const func_approx_data &data) const override;
    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return func_approx_data_ptr_t
     */
    func_approx_data_ptr_t create_approx_data(sym_data &primal, lag_data &raw, shared_data &shared) const override {
        return func_approx_data_ptr_t(make_approx<ipm_constr>(primal, raw, shared));
    }
    DEF_DEFAULT_CLONE(ipm_constr)

};
} // namespace solver
using ipm = solver::ipm_constr;

} // namespace moto

#include <moto/ocp/constr.hpp>

#endif // MOTO_SOLVER_IPM_HPP
