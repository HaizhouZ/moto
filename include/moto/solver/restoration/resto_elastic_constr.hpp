#pragma once

#include <moto/core/fwd.hpp>
#include <moto/solver/linesearch_config.hpp>

namespace moto::solver {

/**
 * @brief Solver-private elastic equality runtime.
 *
 * @details This stores only the coupled local-KKT workspaces associated with
 *   c(w) - p + n = 0.
 * The current multiplier lambda_c is stored in dense dual arrays
 * (__eq_x / __eq_xu) so that the existing equality-dual line-search path can
 * update and back up that state without a second source of truth.
 */
struct resto_elastic_constr {
    size_t ns = 0;
    size_t nc = 0;

    vector p;
    vector p_backup;
    vector d_p;
    vector nu_p;
    vector nu_p_backup;
    vector d_nu_p;

    vector n;
    vector n_backup;
    vector d_n;
    vector nu_n;
    vector nu_n_backup;
    vector d_nu_n;

    vector c_current;
    vector r_c;
    vector r_p;
    vector r_n;
    vector r_s_p;
    vector r_s_n;
    vector combo_p;
    vector combo_n;
    vector b_c;
    vector minv_diag;
    vector minv_bc;
    vector d_lambda;

    void resize(size_t ns_dim, size_t nc_dim);
    size_t dim() const { return ns + nc; }
    void backup_trial_state();
    void restore_trial_state();
    void apply_affine_step(const linesearch_config &cfg);
    void update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary = 0.995) const;
};

} // namespace moto::solver

namespace moto {
using resto_elastic_constr = solver::resto_elastic_constr;
} // namespace moto
