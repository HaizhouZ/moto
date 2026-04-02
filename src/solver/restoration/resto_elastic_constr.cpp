#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>

namespace moto::solver {

namespace {
void resize_zero(vector &v, Eigen::Index n) {
    v.resize(n);
    v.setZero();
}

template <size_t N>
void accumulate_predictor_pairs(const std::array<const vector *, N> &value,
                                const std::array<const vector *, N> &step,
                                const std::array<const vector *, N> &dual,
                                const std::array<const vector *, N> &dual_step,
                                const linesearch_config &cfg,
                                solver::ipm_config::worker &worker) {
    const scalar_t alpha_primal = cfg.alpha_primal;
    const scalar_t alpha_dual = cfg.dual_alpha_for_ineq();
    for (size_t i = 0; i < N; ++i) {
        if (value[i]->size() == 0) {
            continue;
        }
        worker.n_ipm_cstr += static_cast<size_t>(value[i]->size());
        worker.prev_aff_comp += dual[i]->dot(*value[i]);
        worker.post_aff_comp += (*dual[i] + alpha_dual * *dual_step[i]).dot(*value[i] + alpha_primal * *step[i]);
    }
}
} // namespace

void resto_elastic_constr::resize(size_t ns_dim, size_t nc_dim) {
    ns = ns_dim;
    nc = nc_dim;
    const auto dim_eig = static_cast<Eigen::Index>(dim());
    for (auto *v : {&p, &p_backup, &d_p, &nu_p, &nu_p_backup, &d_nu_p,
                    &this->n, &n_backup, &d_n, &nu_n, &nu_n_backup, &d_nu_n,
                    &c_current, &r_c, &r_p, &r_n, &r_s_p, &r_s_n,
                    &combo_p, &combo_n, &b_c, &minv_diag, &minv_bc, &d_lambda}) {
        resize_zero(*v, dim_eig);
    }
}

void resto_elastic_constr::backup_trial_state() {
    positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
}

void resto_elastic_constr::restore_trial_state() {
    positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
}

void resto_elastic_constr::apply_affine_step(const linesearch_config &cfg) {
    positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, cfg.dual_alpha_for_ineq());
    positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, cfg.dual_alpha_for_ineq());
}

void resto_elastic_constr::update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary) const {
    positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
    positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
}

void resto_elastic_constr::finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
    accumulate_predictor_pairs<2>({&p, &n},
                                  {&d_p, &d_n},
                                  {&nu_p, &nu_n},
                                  {&d_nu_p, &d_nu_n},
                                  cfg, worker);
}

scalar_t resto_elastic_constr::penalty_sum() const {
    return p.sum() + n.sum();
}

scalar_t resto_elastic_constr::penalty_dir_deriv() const {
    return d_p.sum() + d_n.sum();
}

scalar_t resto_elastic_constr::barrier_log_sum(scalar_t floor) const {
    if (p.size() == 0) {
        return 0.;
    }
    return p.array().max(floor).log().sum() + n.array().max(floor).log().sum();
}

scalar_t resto_elastic_constr::barrier_dir_deriv(scalar_t floor) const {
    if (p_backup.size() == 0) {
        return 0.;
    }
    return (d_p.array() / p_backup.array().max(floor)).sum() +
           (d_n.array() / n_backup.array().max(floor)).sum();
}

void resto_ineq_constr::resize(size_t nx_dim, size_t nu_dim) {
    nx = nx_dim;
    nu = nu_dim;
    const auto n_dim = static_cast<Eigen::Index>(dim());
    for (auto *v : {&t, &t_backup, &d_t, &nu_t, &nu_t_backup, &d_nu_t,
                    &p, &p_backup, &d_p, &nu_p, &nu_p_backup, &d_nu_p,
                    &n, &n_backup, &d_n, &nu_n, &nu_n_backup, &d_nu_n,
                    &g_current, &r_d, &r_t, &r_p, &r_n, &r_s_t, &r_s_p, &r_s_n,
                    &combo_t, &combo_p, &combo_n, &b_d, &minv_diag, &minv_bd, &d_lambda}) {
        resize_zero(*v, n_dim);
    }
}

void resto_ineq_constr::backup_trial_state() {
    positivity::backup_pair(t, t_backup, nu_t, nu_t_backup);
    positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
}

void resto_ineq_constr::restore_trial_state() {
    positivity::restore_pair(t, t_backup, nu_t, nu_t_backup);
    positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
}

void resto_ineq_constr::apply_affine_step(const linesearch_config &cfg) {
    const scalar_t alpha_dual = cfg.dual_alpha_for_ineq();
    positivity::apply_pair_step(t, d_t, cfg.alpha_primal, nu_t, d_nu_t, alpha_dual);
    positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, alpha_dual);
    positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, alpha_dual);
}

void resto_ineq_constr::update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary) const {
    positivity::update_pair_bounds(cfg, t, d_t, nu_t, d_nu_t, fraction_to_boundary);
    positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
    positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
}

void resto_ineq_constr::finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
    accumulate_predictor_pairs<3>({&t, &p, &n},
                                  {&d_t, &d_p, &d_n},
                                  {&nu_t, &nu_p, &nu_n},
                                  {&d_nu_t, &d_nu_p, &d_nu_n},
                                  cfg, worker);
}

scalar_t resto_ineq_constr::penalty_sum() const {
    return p.sum() + n.sum();
}

scalar_t resto_ineq_constr::penalty_dir_deriv() const {
    return d_p.sum() + d_n.sum();
}

scalar_t resto_ineq_constr::barrier_log_sum(scalar_t floor) const {
    if (t.size() == 0) {
        return 0.;
    }
    return t.array().max(floor).log().sum() +
           p.array().max(floor).log().sum() +
           n.array().max(floor).log().sum();
}

scalar_t resto_ineq_constr::barrier_dir_deriv(scalar_t floor) const {
    if (t_backup.size() == 0) {
        return 0.;
    }
    return (d_t.array() / t_backup.array().max(floor)).sum() +
           (d_p.array() / p_backup.array().max(floor)).sum() +
           (d_n.array() / n_backup.array().max(floor)).sum();
}

} // namespace moto::solver
