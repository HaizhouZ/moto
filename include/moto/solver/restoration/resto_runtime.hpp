#pragma once

#include <moto/core/workspace_data.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>

namespace moto::solver::restoration {

struct prox_data {
    vector u_ref, y_ref;
    vector sigma_u_sq, sigma_y_sq;
};

struct local_residual_info {
    scalar_t stationarity = 0.;
    scalar_t complementarity = 0.;
};

struct reduced_residual_info {
    array_type<row_vector, primal_fields> w_stationarity;
    local_residual_summary eq_local;
    local_residual_summary ineq_local;
    scalar_t inf_primal = 0.;
    scalar_t inf_dual = 0.;
    scalar_t inf_comp = 0.;
};

struct barrier_stats {
    scalar_t avg_comp = 0.;
    scalar_t inf_comp = 0.;
    size_t n_comp = 0;
};

struct objective_summary {
    scalar_t exact_penalty = 0.;
    scalar_t barrier_value = 0.;
    scalar_t penalty_dir_deriv = 0.;
    scalar_t barrier_dir_deriv = 0.;
    scalar_t prim_res_l1 = 0.;
    scalar_t inf_local_stat = 0.;
    scalar_t inf_local_comp = 0.;
};

vector gather_lambda_eq(const ns_riccati::ns_riccati_data &d);
vector gather_lambda_ineq(const ns_riccati::ns_riccati_data &d);
void scatter_lambda_eq(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda);
void scatter_lambda_eq_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda);
void scatter_lambda_ineq(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda);
void scatter_lambda_ineq_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda);
void restore_outer_duals(array_type<vector, constr_fields> &dual,
                         const array_type<vector, constr_fields> &backup);
void commit_bound_state(vector_ref slack,
                        vector_ref multiplier,
                        const vector_const_ref &resto_slack,
                        const vector_const_ref &resto_multiplier,
                        scalar_t threshold,
                        scalar_t reset_value = scalar_t(1.));
bool should_reset_multiplier(const vector_const_ref &multiplier, scalar_t threshold);
void maybe_reset_multiplier(vector_ref multiplier, scalar_t threshold, scalar_t reset_value);
void reset_equality_duals(array_type<vector, constr_fields> &dual, scalar_t threshold);
void reset_equality_duals(ns_riccati::ns_riccati_data &d, scalar_t threshold);
void cleanup_restoration_stage(ns_riccati::ns_riccati_data &d,
                               bool success,
                               scalar_t bound_mult_reset_threshold,
                               scalar_t constr_mult_reset_threshold);
local_residual_info refinement_local_residuals(const ns_riccati::ns_riccati_data::restoration_aux_data &aux);
local_residual_info refinement_local_residuals(const ns_riccati::ns_riccati_data &d);
reduced_residual_info compute_reduced_residual(
    const array_type<row_vector, primal_fields> &lag_jac,
    const array_type<row_vector, primal_fields> &lag_jac_corr,
    const vector_const_ref &dyn_residual,
    const ns_riccati::ns_riccati_data::restoration_aux_data &aux);
reduced_residual_info compute_reduced_residual(const ns_riccati::ns_riccati_data &d);
barrier_stats current_barrier_stats(const ns_riccati::ns_riccati_data::restoration_aux_data &aux);
barrier_stats current_barrier_stats(const ns_riccati::ns_riccati_data &d);
objective_summary current_objective_summary(const ns_riccati::ns_riccati_data::restoration_aux_data &aux);
objective_summary current_objective_summary(const ns_riccati::ns_riccati_data &d);
bool update_mu_bar(ns_riccati::ns_riccati_data::restoration_aux_data &aux,
                   const solver::ipm_config &cfg,
                   scalar_t mu_monotone_fraction_threshold,
                   scalar_t mu_monotone_factor,
                   scalar_t inf_primal,
                   scalar_t inf_dual);
bool update_mu_bar(ns_riccati::ns_riccati_data &d,
                   const solver::ipm_config &cfg,
                   scalar_t mu_monotone_fraction_threshold,
                   scalar_t mu_monotone_factor,
                   scalar_t inf_primal,
                   scalar_t inf_dual);
void load_correction_rhs(array_type<row_vector, primal_fields> &lag_jac_corr,
                         const reduced_residual_info &residual);
void load_correction_rhs(ns_riccati::ns_riccati_data &d, const reduced_residual_info &residual);

void prepare_current_constraint_stack(ns_riccati::ns_riccati_data &d);
void initialize_stage(ns_riccati::ns_riccati_data &d);
void finalize_predictor_step(ns_riccati::ns_riccati_data &d, const linesearch_config &cfg);
void assemble_resto_base_problem(ns_riccati::ns_riccati_data &d,
                                 bool update_derivatives,
                                 scalar_t rho_u,
                                 scalar_t rho_y,
                                 vector *u_diag = nullptr,
                                 vector *y_diag = nullptr);
void finalize_newton_step(ns_riccati::ns_riccati_data &d);
void update_ls_bounds(ns_riccati::ns_riccati_data &d, workspace_data *cfg);
void backup_trial_state(ns_riccati::ns_riccati_data &d);
void restore_trial_state(ns_riccati::ns_riccati_data &d);
void apply_affine_step(ns_riccati::ns_riccati_data &d, workspace_data *cfg);

} // namespace moto::solver::restoration
