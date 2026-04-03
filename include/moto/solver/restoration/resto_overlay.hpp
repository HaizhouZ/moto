#pragma once

#include <moto/ocp/cost.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/core/array.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/linesearch_config.hpp>

#include <array>

namespace moto {
struct node_data;
namespace solver::ns_riccati {
struct ns_riccati_data;
}
}

namespace moto::solver::restoration {

namespace detail {

enum elastic_slot_t : size_t {
    slot_t = 0,
    slot_p = 1,
    slot_n = 2,
};

template <typename T>
using elastic_pair_array = array_type<T, std::array{slot_p, slot_n}>;

template <typename T>
using elastic_triplet_array = array_type<T, std::array{slot_t, slot_p, slot_n}>;

struct eq_local_state {
    size_t ns = 0;
    size_t nc = 0;

    elastic_pair_array<vector> value;
    elastic_pair_array<vector> value_backup;
    elastic_pair_array<vector> d_value;
    elastic_pair_array<vector> dual;
    elastic_pair_array<vector> dual_backup;
    elastic_pair_array<vector> d_dual;
    elastic_pair_array<vector> r_stat;
    elastic_pair_array<vector> r_comp;
    elastic_pair_array<vector> combo;
    elastic_pair_array<vector> corrector;
    vector base_residual, r_c, b_c, minv_diag, minv_bc, d_lambda;
};

struct ineq_local_state {
    size_t nx = 0;
    size_t nu = 0;

    elastic_triplet_array<vector> value;
    elastic_triplet_array<vector> value_backup;
    elastic_triplet_array<vector> d_value;
    elastic_triplet_array<vector> dual;
    elastic_triplet_array<vector> dual_backup;
    elastic_triplet_array<vector> d_dual;
    elastic_triplet_array<vector> r_comp;
    elastic_triplet_array<vector> denom;
    elastic_triplet_array<vector> combo;
    elastic_triplet_array<vector> corrector;
    elastic_pair_array<vector> r_stat;
    vector base_residual, r_d, b_d, minv_diag, minv_bd;
};

} // namespace detail

struct elastic_init_pair {
    scalar_t p = 0.;
    scalar_t n = 0.;
    scalar_t z_p = 0.;
    scalar_t z_n = 0.;
    scalar_t lambda = 0.;
    scalar_t weight = 0.;
};

struct elastic_init_ineq_scalar {
    scalar_t t = 0.;
    scalar_t p = 0.;
    scalar_t n = 0.;
    scalar_t nu_t = 0.;
    scalar_t nu_p = 0.;
    scalar_t nu_n = 0.;
};

struct local_residual_summary {
    scalar_t inf_prim = 0.;
    scalar_t inf_stat = 0.;
    scalar_t inf_comp = 0.;
};

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
        vector multiplier_backup;
        detail::eq_local_state elastic;
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
    static elastic_init_pair initialize_elastic_pair(scalar_t c, scalar_t rho, scalar_t mu_bar);
    static void compute_local_model(detail::eq_local_state &elastic,
                                    const vector_const_ref &base_residual,
                                    const vector_const_ref &multiplier,
                                    scalar_t rho,
                                    scalar_t mu_bar,
                                    const vector *mu_p_target,
                                    const vector *mu_n_target,
                                    scalar_t lambda_reg);
    static void compute_local_model(detail::eq_local_state &elastic,
                                    const vector_const_ref &base_residual,
                                    const vector_const_ref &multiplier,
                                    scalar_t rho,
                                    scalar_t mu_bar,
                                    scalar_t lambda_reg);
    static void recover_local_step(const vector_const_ref &delta_c,
                                   detail::eq_local_state &elastic,
                                   scalar_t lambda_reg);
    static local_residual_summary current_local_residuals(const detail::eq_local_state &elastic);
    static local_residual_summary linearized_newton_residuals(const vector_const_ref &delta_c,
                                                              const detail::eq_local_state &elastic,
                                                              scalar_t lambda_reg);
    const constr &source() const { return source_; }
    field_t source_field() const { return source_->field(); }
    size_t source_pos() const { return source_pos_; }

    DEF_DEFAULT_CLONE(resto_eq_elastic_constr)

  private:
    static void resize_local_state(detail::eq_local_state &state, size_t ns_dim, size_t nc_dim);
    void compute_local_model(approx_data &data,
                             scalar_t mu_bar,
                             const vector *mu_p_target = nullptr,
                             const vector *mu_n_target = nullptr) const;
    local_residual_summary current_local_residuals(const approx_data &data) const;

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
        vector multiplier_backup;
        detail::ineq_local_state elastic;
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
    static elastic_init_ineq_scalar initialize_elastic_ineq_scalar(scalar_t g,
                                                                   scalar_t rho,
                                                                   scalar_t mu_bar,
                                                                   scalar_t nu_t_init);
    static void compute_local_model(detail::ineq_local_state &elastic,
                                    const vector_const_ref &base_residual,
                                    scalar_t rho,
                                    scalar_t mu_bar,
                                    const vector *mu_t_target,
                                    const vector *mu_p_target,
                                    const vector *mu_n_target,
                                    scalar_t lambda_reg);
    static void compute_local_model(detail::ineq_local_state &elastic,
                                    const vector_const_ref &base_residual,
                                    scalar_t rho,
                                    scalar_t mu_bar,
                                    scalar_t lambda_reg);
    static void recover_local_step(const vector_const_ref &delta_g,
                                   detail::ineq_local_state &ineq,
                                   scalar_t lambda_reg);
    static local_residual_summary current_local_residuals(const detail::ineq_local_state &ineq);
    static local_residual_summary linearized_newton_residuals(const vector_const_ref &delta_g,
                                                              const detail::ineq_local_state &ineq,
                                                              scalar_t lambda_reg);
    const constr &source() const { return source_; }
    field_t source_field() const { return source_->field(); }
    size_t source_pos() const { return source_pos_; }

    DEF_DEFAULT_CLONE(resto_ineq_elastic_ipm_constr)

  private:
    static void resize_local_state(detail::ineq_local_state &state, size_t nx_dim, size_t nu_dim);
    void compute_local_model(approx_data &data,
                             scalar_t mu_bar,
                             const vector *mu_t_target = nullptr,
                             const vector *mu_p_target = nullptr,
                             const vector *mu_n_target = nullptr) const;
    local_residual_summary current_local_residuals(const approx_data &data) const;

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
void restore_outer_duals(array_type<vector, constr_fields> &dual,
                         const array_type<vector, constr_fields> &backup);
void commit_bound_state(vector_ref slack,
                        vector_ref multiplier,
                        const vector_const_ref &resto_slack,
                        const vector_const_ref &resto_multiplier,
                        scalar_t reset_threshold,
                        scalar_t reset_value);

bool should_reset_multiplier(const vector_const_ref &multiplier, scalar_t threshold);
void maybe_reset_multiplier(vector_ref multiplier, scalar_t threshold, scalar_t reset_value);
void reset_equality_duals(array_type<vector, constr_fields> &dual, scalar_t threshold);
void reset_equality_duals(ns_riccati::ns_riccati_data &d, scalar_t threshold);

} // namespace moto::solver::restoration
