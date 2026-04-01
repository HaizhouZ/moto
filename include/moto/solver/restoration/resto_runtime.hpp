#pragma once

#include <moto/core/workspace_data.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto::solver::restoration {

vector gather_lambda(const ns_riccati::ns_riccati_data &d);
void scatter_lambda(ns_riccati::ns_riccati_data &d, const vector_const_ref &lambda);
void scatter_lambda_step(ns_riccati::ns_riccati_data &d, const vector_const_ref &delta_lambda);

void initialize_stage(ns_riccati::ns_riccati_data &d);
void finalize_newton_step(ns_riccati::ns_riccati_data &d);
void update_ls_bounds(ns_riccati::ns_riccati_data &d, workspace_data *cfg);
void backup_trial_state(ns_riccati::ns_riccati_data &d);
void restore_trial_state(ns_riccati::ns_riccati_data &d);
void apply_affine_step(ns_riccati::ns_riccati_data &d, workspace_data *cfg);

} // namespace moto::solver::restoration
