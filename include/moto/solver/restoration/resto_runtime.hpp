#pragma once

#include <moto/core/workspace_data.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto::solver::restoration {

struct prox_data {
    vector u_ref, y_ref;
    vector sigma_u_sq, sigma_y_sq;
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

void prepare_current_constraint_stack(ns_riccati::ns_riccati_data &d);
void initialize_stage(ns_riccati::ns_riccati_data &d);
void assemble_resto_base_problem(ns_riccati::ns_riccati_data &d,
                                 bool update_derivatives,
                                 scalar_t rho_u,
                                 scalar_t rho_y,
                                 aligned_vector_map_t *u_diag = nullptr,
                                 aligned_vector_map_t *y_diag = nullptr);
void finalize_newton_step(ns_riccati::ns_riccati_data &d);
void update_ls_bounds(ns_riccati::ns_riccati_data &d, workspace_data *cfg);
void backup_trial_state(ns_riccati::ns_riccati_data &d);
void restore_trial_state(ns_riccati::ns_riccati_data &d);
void apply_affine_step(ns_riccati::ns_riccati_data &d, workspace_data *cfg);

} // namespace moto::solver::restoration
