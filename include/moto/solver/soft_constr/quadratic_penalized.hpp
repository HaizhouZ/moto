#pragma once
#include <moto/ocp/soft_constr.hpp>

namespace moto {
namespace solver {
/// @brief Soft equality constraint via proximal augmented Lagrangian (no slack):
///   L = lambda^T*C(x) - rho/2*||delta lambda||^2
/// Stagewise KKT over (du, dlam):
///   [L_Hess  J^T  ] [du  ]   [L_jac]
///   [J      -rho*I] [dlam] = -[h    ]
/// Gradient contribution (Schur complement): +(1/rho)*J^T*h added to L_jac
///   node_data adds J^T*lambda to Q_u; jac_modification adds (h/rho - lambda)^T*J to cancel it
/// Hessian contribution (Schur complement):  (1/rho) * J^T * J
/// Dual update from row 2: dlam = (J*du + h) / rho
/// As rho -> 0: (1/rho)*J^T*J dominates, forcing J*du = -h (hard constraint recovery).
/// As rho -> inf: regularization vanishes, soft constraint has negligible effect.
class pmm_constr final : public soft_constr {
  private:
    using base = soft_constr;

  public:
    struct approx_data : public base::approx_data {
        vector g_;                  ///< raw constraint value C(x) = h
        vector multiplier_backup_;  ///< backup of multiplier for line search trials
        scalar_t rho_ = 1.0;        ///< dual penalty weight (copied from constraint at construction)

        approx_data(base::approx_data &&rhs, scalar_t rho);
    };

    scalar_t rho = 1.0; ///< dual penalty weight

    using base::base;

    void value_impl(func_approx_data &data) const override final;
    void jacobian_impl(func_approx_data &data) const override final;

    void propagate_jacobian(func_approx_data &d) const override;
    void propagate_hessian(func_approx_data &d) const override;
    void propagate_res_stats(func_approx_data &d) const override {};

    /// @brief initialize: set lambda = 0 (cold start)
    void initialize(data_map_t &data) const override final;
    /// @brief compute d_multiplier = (J*du + h) / rho  from row 2 of KKT
    void finalize_newton_step(data_map_t &data) const override final;
    /// @brief no-op: no predictor step
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) const override final {};
    /// @brief backup slack and multiplier before a line-search attempt
    void backup_trial_state(data_map_t &data) const override final;
    /// @brief restore slack and multiplier for the next line-search attempt
    void restore_trial_state(data_map_t &data) const override final;
    /// @brief apply: lambda += alpha*d_multiplier
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override final;

    func_approx_data_ptr_t create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const override {
        std::unique_ptr<base::approx_data> base_d(make_approx<soft_constr>(primal, raw, shared));
        return func_approx_data_ptr_t(new approx_data(std::move(*base_d), rho));
    }
    DEF_DEFAULT_CLONE(pmm_constr)
};
} // namespace solver
using pmm_constr = solver::pmm_constr;
} // namespace moto
