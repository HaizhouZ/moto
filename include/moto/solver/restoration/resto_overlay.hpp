#pragma once

#include <moto/ocp/cost.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>

namespace moto {
struct node_data;
}

namespace moto::solver::restoration {

struct restoration_overlay_settings {
    scalar_t rho_u = 1e-4;
    scalar_t rho_y = 1e-4;
    scalar_t rho_eq = 1000.0;
    scalar_t rho_ineq = 1000.0;
    scalar_t lambda_reg = 1e-8;
};

class resto_prox_cost final : public generic_cost {
  public:
    struct approx_data : public func_approx_data {
        vector u_ref;
        vector y_ref;
        vector sigma_u_sq;
        vector sigma_y_sq;

        approx_data(sym_data &primal, lag_data &raw, shared_data &shared, const generic_func &f)
            : func_approx_data(primal, raw, shared, f) {}
    };

    resto_prox_cost(const std::string &name,
                    const var_list &u_args,
                    const var_list &y_args);

    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              lag_data &raw,
                                              shared_data &shared) const override;

    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void hessian_impl(func_approx_data &data) const override;

    DEF_DEFAULT_CLONE(resto_prox_cost)
};

class resto_eq_elastic_constr final : public soft_constr {
  public:
    struct approx_data : public soft_constr::approx_data {
        solver::ipm_config *ipm_cfg = nullptr;
        vector base_residual;
        resto_elastic_constr elastic;
        scalar_t rho = 1000.0;
        scalar_t lambda_reg = 1e-8;

        explicit approx_data(soft_constr::approx_data &&rhs)
            : soft_constr::approx_data(std::move(rhs)) {}
    };

    resto_eq_elastic_constr(const std::string &name,
                            const constr &source,
                            size_t source_pos,
                            scalar_t rho,
                            scalar_t lambda_reg);

    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override;
    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              lag_data &raw,
                                              shared_data &shared) const override;

    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void hessian_impl(func_approx_data &data) const override;
    void propagate_jacobian(func_approx_data &data) const override;
    void propagate_hessian(func_approx_data &data) const override;
    void propagate_res_stats(func_approx_data &data) const override;

    void initialize(data_map_t &data) const override;
    void finalize_newton_step(data_map_t &data) const override;
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) const override;
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override;
    void update_ls_bounds(data_map_t &data, workspace_data *cfg) const override;
    void backup_trial_state(data_map_t &data) const override;
    void restore_trial_state(data_map_t &data) const override;
    scalar_t objective_penalty(const func_approx_data &data) const override;
    scalar_t objective_penalty_dir_deriv(const func_approx_data &data) const override;
    scalar_t search_penalty(const func_approx_data &data) const override;
    scalar_t search_penalty_dir_deriv(const func_approx_data &data) const override;
    scalar_t local_stat_residual_inf(const func_approx_data &data) const override;
    scalar_t local_comp_residual_inf(const func_approx_data &data) const override;

    const constr &source() const { return source_; }
    field_t source_field() const { return source_->field(); }
    size_t source_pos() const { return source_pos_; }

    DEF_DEFAULT_CLONE(resto_eq_elastic_constr)

  private:
    constr source_;
    size_t source_pos_ = 0;
    scalar_t rho_ = 1000.0;
    scalar_t lambda_reg_ = 1e-8;
};

class resto_ineq_elastic_ipm_constr final : public ineq_constr {
  public:
    struct approx_data : public ineq_constr::approx_data {
        solver::ipm_config *ipm_cfg = nullptr;
        vector base_residual;
        resto_ineq_constr elastic;
        scalar_t rho = 1000.0;
        scalar_t lambda_reg = 1e-8;

        explicit approx_data(ineq_constr::approx_data &&rhs)
            : ineq_constr::approx_data(std::move(rhs)) {}
    };

    resto_ineq_elastic_ipm_constr(const std::string &name,
                                  const constr &source,
                                  size_t source_pos,
                                  scalar_t rho,
                                  scalar_t lambda_reg);

    void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const override;
    func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                              lag_data &raw,
                                              shared_data &shared) const override;

    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;
    void hessian_impl(func_approx_data &data) const override;
    void propagate_jacobian(func_approx_data &data) const override;
    void propagate_hessian(func_approx_data &data) const override;
    void propagate_res_stats(func_approx_data &data) const override;

    void initialize(data_map_t &data) const override;
    void finalize_newton_step(data_map_t &data) const override;
    void finalize_predictor_step(data_map_t &data, workspace_data *cfg) const override;
    void apply_corrector_step(data_map_t &data) const override;
    void apply_affine_step(data_map_t &data, workspace_data *cfg) const override;
    void update_ls_bounds(data_map_t &data, workspace_data *cfg) const override;
    void backup_trial_state(data_map_t &data) const override;
    void restore_trial_state(data_map_t &data) const override;
    scalar_t objective_penalty(const func_approx_data &data) const override;
    scalar_t objective_penalty_dir_deriv(const func_approx_data &data) const override;
    scalar_t search_penalty(const func_approx_data &data) const override;
    scalar_t search_penalty_dir_deriv(const func_approx_data &data) const override;
    scalar_t local_stat_residual_inf(const func_approx_data &data) const override;
    scalar_t local_comp_residual_inf(const func_approx_data &data) const override;

    const constr &source() const { return source_; }
    field_t source_field() const { return source_->field(); }
    size_t source_pos() const { return source_pos_; }

    DEF_DEFAULT_CLONE(resto_ineq_elastic_ipm_constr)

  private:
    constr source_;
    size_t source_pos_ = 0;
    scalar_t rho_ = 1000.0;
    scalar_t lambda_reg_ = 1e-8;
};

ocp_ptr_t build_restoration_overlay_problem(const ocp_ptr_t &source_prob,
                                            const restoration_overlay_settings &settings);

void sync_restoration_overlay_primal(node_data &outer, node_data &resto);
void sync_restoration_overlay_duals(node_data &outer, node_data &resto);
void seed_restoration_overlay_refs(node_data &resto,
                                   scalar_t prox_eps = scalar_t(1.0));

} // namespace moto::solver::restoration
