#pragma once

#include <moto/core/fwd.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/linesearch_config.hpp>

namespace moto::solver {

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
    void update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary = scalar_t(0.995)) const;
    void finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const;
    scalar_t penalty_sum() const;
    scalar_t penalty_dir_deriv() const;
    scalar_t barrier_log_sum(scalar_t floor = scalar_t(1e-16)) const;
    scalar_t barrier_dir_deriv(scalar_t floor = scalar_t(1e-16)) const;
};

struct resto_ineq_constr {
    size_t nx = 0;
    size_t nu = 0;

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

    vector g_current;
    vector r_d;
    vector r_p;
    vector r_n;
    vector r_s_p;
    vector r_s_n;
    vector combo_p;
    vector combo_n;
    vector b_d;
    vector minv_diag;
    vector minv_bd;
    vector d_lambda;

    void resize(size_t nx_dim, size_t nu_dim);
    size_t dim() const { return nx + nu; }
    void backup_trial_state();
    void restore_trial_state();
    void apply_affine_step(const linesearch_config &cfg);
    void update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary = scalar_t(0.995)) const;
    void finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const;
    scalar_t penalty_sum() const;
    scalar_t penalty_dir_deriv() const;
    scalar_t barrier_log_sum(scalar_t floor = scalar_t(1e-16)) const;
    scalar_t barrier_dir_deriv(scalar_t floor = scalar_t(1e-16)) const;
};

} // namespace moto::solver

namespace moto {
using resto_elastic_constr = solver::resto_elastic_constr;
using resto_ineq_constr = solver::resto_ineq_constr;
} // namespace moto

namespace moto::solver {
namespace {
inline void resize_zero(vector &v, Eigen::Index n) {
    v.resize(n);
    v.setZero();
}

template <size_t N>
inline void accumulate_predictor_pairs(const std::array<const vector *, N> &value,
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

inline void resto_elastic_constr::resize(size_t ns_dim, size_t nc_dim) {
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

inline void resto_elastic_constr::backup_trial_state() {
    positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
}

inline void resto_elastic_constr::restore_trial_state() {
    positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
}

inline void resto_elastic_constr::apply_affine_step(const linesearch_config &cfg) {
    positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, cfg.dual_alpha_for_ineq());
    positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, cfg.dual_alpha_for_ineq());
}

inline void resto_elastic_constr::update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary) const {
    positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
    positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
}

inline void resto_elastic_constr::finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
    accumulate_predictor_pairs<2>({&p, &n}, {&d_p, &d_n}, {&nu_p, &nu_n}, {&d_nu_p, &d_nu_n}, cfg, worker);
}

inline scalar_t resto_elastic_constr::penalty_sum() const { return p.sum() + n.sum(); }
inline scalar_t resto_elastic_constr::penalty_dir_deriv() const { return d_p.sum() + d_n.sum(); }

inline scalar_t resto_elastic_constr::barrier_log_sum(scalar_t floor) const {
    if (p.size() == 0) {
        return 0.;
    }
    return p.array().max(floor).log().sum() + n.array().max(floor).log().sum();
}

inline scalar_t resto_elastic_constr::barrier_dir_deriv(scalar_t floor) const {
    if (p_backup.size() == 0) {
        return 0.;
    }
    return (d_p.array() / p_backup.array().max(floor)).sum() +
           (d_n.array() / n_backup.array().max(floor)).sum();
}

inline void resto_ineq_constr::resize(size_t nx_dim, size_t nu_dim) {
    nx = nx_dim;
    nu = nu_dim;
    const auto n_dim = static_cast<Eigen::Index>(dim());
    for (auto *v : {&p, &p_backup, &d_p, &nu_p, &nu_p_backup, &d_nu_p,
                    &n, &n_backup, &d_n, &nu_n, &nu_n_backup, &d_nu_n,
                    &g_current, &r_d, &r_p, &r_n, &r_s_p, &r_s_n,
                    &combo_p, &combo_n, &b_d, &minv_diag, &minv_bd, &d_lambda}) {
        resize_zero(*v, n_dim);
    }
}

inline void resto_ineq_constr::backup_trial_state() {
    positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
}

inline void resto_ineq_constr::restore_trial_state() {
    positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
    positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
}

inline void resto_ineq_constr::apply_affine_step(const linesearch_config &cfg) {
    const scalar_t alpha_dual = cfg.dual_alpha_for_ineq();
    positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, alpha_dual);
    positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, alpha_dual);
}

inline void resto_ineq_constr::update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary) const {
    positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
    positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
}

inline void resto_ineq_constr::finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
    accumulate_predictor_pairs<2>({&p, &n}, {&d_p, &d_n}, {&nu_p, &nu_n}, {&d_nu_p, &d_nu_n}, cfg, worker);
}

inline scalar_t resto_ineq_constr::penalty_sum() const { return p.sum() + n.sum(); }
inline scalar_t resto_ineq_constr::penalty_dir_deriv() const { return d_p.sum() + d_n.sum(); }

inline scalar_t resto_ineq_constr::barrier_log_sum(scalar_t floor) const {
    if (p.size() == 0) {
        return 0.;
    }
    return p.array().max(floor).log().sum() + n.array().max(floor).log().sum();
}

inline scalar_t resto_ineq_constr::barrier_dir_deriv(scalar_t floor) const {
    if (p_backup.size() == 0) {
        return 0.;
    }
    return (d_p.array() / p_backup.array().max(floor)).sum() +
           (d_n.array() / n_backup.array().max(floor)).sum();
}
} // namespace moto::solver
