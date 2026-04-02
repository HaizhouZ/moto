#pragma once

#include <moto/core/fwd.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/linesearch_config.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace moto::solver::restoration {

namespace detail {

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

struct eq_local_state {
    size_t ns = 0;
    size_t nc = 0;

    vector p, p_backup, d_p, nu_p, nu_p_backup, d_nu_p;
    vector n, n_backup, d_n, nu_n, nu_n_backup, d_nu_n;
    vector c_current, r_c, r_p, r_n, r_s_p, r_s_n;
    vector combo_p, combo_n, b_c, minv_diag, minv_bc, d_lambda;
    vector corrector_p, corrector_n;

    void resize(size_t ns_dim, size_t nc_dim) {
        ns = ns_dim;
        nc = nc_dim;
        const auto dim_eig = static_cast<Eigen::Index>(ns + nc);
        for (auto *v : {&p, &p_backup, &d_p, &nu_p, &nu_p_backup, &d_nu_p,
                        &n, &n_backup, &d_n, &nu_n, &nu_n_backup, &d_nu_n,
                        &c_current, &r_c, &r_p, &r_n, &r_s_p, &r_s_n,
                        &combo_p, &combo_n, &b_c, &minv_diag, &minv_bc, &d_lambda,
                        &corrector_p, &corrector_n}) {
            resize_zero(*v, dim_eig);
        }
    }

    size_t dim() const { return ns + nc; }

    void backup_trial_state() {
        positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
        positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
    }

    void restore_trial_state() {
        positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
        positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
    }

    void apply_affine_step(const linesearch_config &cfg) {
        positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, cfg.dual_alpha_for_ineq());
        positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, cfg.dual_alpha_for_ineq());
    }

    void update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary = scalar_t(0.995)) const {
        positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
        positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
    }

    void finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
        accumulate_predictor_pairs<2>({&p, &n}, {&d_p, &d_n}, {&nu_p, &nu_n}, {&d_nu_p, &d_nu_n}, cfg, worker);
    }

    scalar_t penalty_sum() const { return p.sum() + n.sum(); }
    scalar_t penalty_dir_deriv() const { return d_p.sum() + d_n.sum(); }

    scalar_t barrier_log_sum(scalar_t floor = scalar_t(1e-16)) const {
        if (p.size() == 0) {
            return 0.;
        }
        return p.array().max(floor).log().sum() + n.array().max(floor).log().sum();
    }

    scalar_t barrier_dir_deriv(scalar_t floor = scalar_t(1e-16)) const {
        if (p_backup.size() == 0) {
            return 0.;
        }
        return (d_p.array() / p_backup.array().max(floor)).sum() +
               (d_n.array() / n_backup.array().max(floor)).sum();
    }
};

struct ineq_local_state {
    size_t nx = 0;
    size_t nu = 0;

    vector t, t_backup, d_t, nu_t, nu_t_backup, d_nu_t;
    vector p, p_backup, d_p, nu_p, nu_p_backup, d_nu_p;
    vector n, n_backup, d_n, nu_n, nu_n_backup, d_nu_n;
    vector g_current, r_d, r_p, r_n, r_s_t, r_s_p, r_s_n;
    vector denom_t, denom_p, denom_n;
    vector combo_t, combo_p, combo_n, b_d, minv_diag, minv_bd;
    vector corrector_t, corrector_p, corrector_n;

    void resize(size_t nx_dim, size_t nu_dim) {
        nx = nx_dim;
        nu = nu_dim;
        const auto dim_eig = static_cast<Eigen::Index>(nx + nu);
        for (auto *v : {&t, &t_backup, &d_t, &nu_t, &nu_t_backup, &d_nu_t,
                        &p, &p_backup, &d_p, &nu_p, &nu_p_backup, &d_nu_p,
                        &n, &n_backup, &d_n, &nu_n, &nu_n_backup, &d_nu_n,
                        &g_current, &r_d, &r_p, &r_n, &r_s_t, &r_s_p, &r_s_n,
                        &denom_t, &denom_p, &denom_n,
                        &combo_t, &combo_p, &combo_n, &b_d, &minv_diag, &minv_bd,
                        &corrector_t, &corrector_p, &corrector_n}) {
            resize_zero(*v, dim_eig);
        }
    }

    size_t dim() const { return nx + nu; }

    void backup_trial_state() {
        positivity::backup_pair(t, t_backup, nu_t, nu_t_backup);
        positivity::backup_pair(p, p_backup, nu_p, nu_p_backup);
        positivity::backup_pair(n, n_backup, nu_n, nu_n_backup);
    }

    void restore_trial_state() {
        positivity::restore_pair(t, t_backup, nu_t, nu_t_backup);
        positivity::restore_pair(p, p_backup, nu_p, nu_p_backup);
        positivity::restore_pair(n, n_backup, nu_n, nu_n_backup);
    }

    void apply_affine_step(const linesearch_config &cfg) {
        positivity::apply_pair_step(t, d_t, cfg.alpha_primal, nu_t, d_nu_t, cfg.dual_alpha_for_ineq());
        positivity::apply_pair_step(p, d_p, cfg.alpha_primal, nu_p, d_nu_p, cfg.dual_alpha_for_ineq());
        positivity::apply_pair_step(n, d_n, cfg.alpha_primal, nu_n, d_nu_n, cfg.dual_alpha_for_ineq());
    }

    void update_ls_bounds(linesearch_config &cfg, scalar_t fraction_to_boundary = scalar_t(0.995)) const {
        positivity::update_pair_bounds(cfg, t, d_t, nu_t, d_nu_t, fraction_to_boundary);
        positivity::update_pair_bounds(cfg, p, d_p, nu_p, d_nu_p, fraction_to_boundary);
        positivity::update_pair_bounds(cfg, n, d_n, nu_n, d_nu_n, fraction_to_boundary);
    }

    void finalize_predictor_step(const linesearch_config &cfg, solver::ipm_config::worker &worker) const {
        accumulate_predictor_pairs<3>({&t, &p, &n}, {&d_t, &d_p, &d_n}, {&nu_t, &nu_p, &nu_n}, {&d_nu_t, &d_nu_p, &d_nu_n}, cfg, worker);
    }

    scalar_t penalty_sum() const { return p.sum() + n.sum(); }
    scalar_t penalty_dir_deriv() const { return d_p.sum() + d_n.sum(); }

    scalar_t barrier_log_sum(scalar_t floor = scalar_t(1e-16)) const {
        if (p.size() == 0) {
            return 0.;
        }
        return t.array().max(floor).log().sum() +
               p.array().max(floor).log().sum() +
               n.array().max(floor).log().sum();
    }

    scalar_t barrier_dir_deriv(scalar_t floor = scalar_t(1e-16)) const {
        if (p_backup.size() == 0) {
            return 0.;
        }
        return (d_t.array() / t_backup.array().max(floor)).sum() +
               (d_p.array() / p_backup.array().max(floor)).sum() +
               (d_n.array() / n_backup.array().max(floor)).sum();
    }
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

inline elastic_init_pair initialize_elastic_pair(scalar_t c, scalar_t rho, scalar_t mu_bar) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires mu_bar > 0");
    }

    const scalar_t disc = (mu_bar - rho * c) * (mu_bar - rho * c) + scalar_t(2.) * rho * mu_bar * c;
    const scalar_t sqrt_disc = std::sqrt(std::max(disc, scalar_t(0.)));
    const scalar_t n = (mu_bar - rho * c + sqrt_disc) / (scalar_t(2.) * rho);
    const scalar_t p = c + n;

    elastic_init_pair out;
    out.p = std::max(p, scalar_t(1e-16));
    out.n = std::max(n, scalar_t(1e-16));
    out.z_p = mu_bar / out.p;
    out.z_n = mu_bar / out.n;
    out.lambda = rho - out.z_p;
    const scalar_t a = out.z_p / out.p;
    const scalar_t b = out.z_n / out.n;
    out.weight = (a > 0. && b > 0.) ? (a * b) / (a + b) : scalar_t(0.);
    return out;
}

struct elastic_init_ineq_scalar {
    scalar_t t = 0.;
    scalar_t p = 0.;
    scalar_t n = 0.;
    scalar_t nu_t = 0.;
    scalar_t nu_p = 0.;
    scalar_t nu_n = 0.;
};

inline elastic_init_ineq_scalar initialize_elastic_ineq_scalar(scalar_t g, scalar_t rho, scalar_t mu_bar) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires mu_bar > 0");
    }

    elastic_init_ineq_scalar out;
    const scalar_t eps = 1e-12;
    const auto f = [&](scalar_t nu_t) {
        return g + mu_bar / nu_t - mu_bar / (rho - nu_t) + mu_bar / (rho + nu_t);
    };
    scalar_t lo = eps;
    scalar_t hi = rho - eps;
    for (int iter = 0; iter < 100; ++iter) {
        const scalar_t mid = scalar_t(0.5) * (lo + hi);
        if (f(mid) > 0.) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out.nu_t = scalar_t(0.5) * (lo + hi);
    out.nu_p = rho - out.nu_t;
    out.nu_n = rho + out.nu_t;
    out.t = mu_bar / out.nu_t;
    out.p = mu_bar / out.nu_p;
    out.n = mu_bar / out.nu_n;
    return out;
}

struct local_residual_summary {
    scalar_t inf_prim = 0.;
    scalar_t inf_stat = 0.;
    scalar_t inf_comp = 0.;
};

inline void compute_local_model(const vector_const_ref &c,
                                const vector_const_ref &lambda,
                                detail::eq_local_state &elastic,
                                scalar_t rho,
                                scalar_t mu_bar,
                                const vector *mu_p_target,
                                const vector *mu_n_target,
                                scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model requires lambda_reg >= 0");
    }
    if (c.size() != lambda.size() ||
        c.size() != elastic.p.size() ||
        c.size() != elastic.n.size() ||
        c.size() != elastic.nu_p.size() ||
        c.size() != elastic.nu_n.size()) {
        throw std::runtime_error("compute_local_model size mismatch");
    }

    elastic.c_current = c;
    elastic.r_c.noalias() = c - elastic.p + elastic.n;
    elastic.r_p = vector::Constant(c.size(), rho) - lambda - elastic.nu_p;
    elastic.r_n = vector::Constant(c.size(), rho) + lambda - elastic.nu_n;
    if (mu_p_target != nullptr && mu_p_target->size() != c.size()) {
        throw std::runtime_error("compute_local_model mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != c.size()) {
        throw std::runtime_error("compute_local_model mu_n_target size mismatch");
    }
    for (Eigen::Index i = 0; i < c.size(); ++i) {
        const scalar_t mu_p = mu_p_target ? (*mu_p_target)(i) : mu_bar;
        const scalar_t mu_n = mu_n_target ? (*mu_n_target)(i) : mu_bar;
        elastic.r_s_p(i) = elastic.nu_p(i) * elastic.p(i) - mu_p;
        elastic.r_s_n(i) = elastic.nu_n(i) * elastic.n(i) - mu_n;
    }

    if (lambda_reg == 0.) {
        const auto denom_p = elastic.nu_p.array().max(eps);
        const auto denom_n = elastic.nu_n.array().max(eps);
        elastic.combo_p =
            (elastic.r_s_p.array() + elastic.p.array() * elastic.r_p.array()) / denom_p;
        elastic.combo_n =
            (elastic.r_s_n.array() + elastic.n.array() * elastic.r_n.array()) / denom_n;
        elastic.minv_diag =
            (elastic.p.array() / denom_p + elastic.n.array() / denom_n).inverse();
    } else {
        const auto denom_p = (elastic.p.array() + lambda_reg * elastic.nu_p.array()).max(eps);
        const auto denom_n = (elastic.n.array() + lambda_reg * elastic.nu_n.array()).max(eps);
        elastic.combo_p =
            (elastic.p.array() * elastic.r_p.array() + lambda_reg * elastic.r_s_p.array()) / denom_p;
        elastic.combo_n =
            (elastic.n.array() * elastic.r_n.array() + lambda_reg * elastic.r_s_n.array()) / denom_n;
        elastic.minv_diag =
            (elastic.p.array() / denom_p + elastic.n.array() / denom_n).inverse();
    }

    elastic.b_c = elastic.r_c + elastic.combo_p - elastic.combo_n;
    elastic.minv_bc = elastic.minv_diag.array() * elastic.b_c.array();
}

inline void compute_local_model(const vector_const_ref &c,
                                const vector_const_ref &lambda,
                                detail::eq_local_state &elastic,
                                scalar_t rho,
                                scalar_t mu_bar,
                                scalar_t lambda_reg) {
    compute_local_model(c, lambda, elastic, rho, mu_bar, nullptr, nullptr, lambda_reg);
}

inline void recover_local_step(const vector_const_ref &delta_c,
                               detail::eq_local_state &elastic,
                               scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (delta_c.size() != elastic.dim()) {
        throw std::runtime_error("recover_local_step size mismatch");
    }
    elastic.d_lambda = elastic.minv_diag.array() * (delta_c.array() + elastic.b_c.array());
    if (lambda_reg == 0.) {
        elastic.d_nu_p = elastic.r_p - elastic.d_lambda;
        elastic.d_nu_n = elastic.r_n + elastic.d_lambda;
        elastic.d_p =
            -(elastic.r_s_p.array() + elastic.p.array() * elastic.d_nu_p.array()) /
            elastic.nu_p.array().max(eps);
        elastic.d_n =
            -(elastic.r_s_n.array() + elastic.n.array() * elastic.d_nu_n.array()) /
            elastic.nu_n.array().max(eps);
    } else {
        const auto denom_p = (elastic.p.array() + lambda_reg * elastic.nu_p.array()).max(eps);
        const auto denom_n = (elastic.n.array() + lambda_reg * elastic.nu_n.array()).max(eps);
        elastic.d_nu_p =
            (elastic.nu_p.array() * (elastic.r_p.array() - elastic.d_lambda.array()) - elastic.r_s_p.array()) /
            denom_p;
        elastic.d_nu_n =
            (elastic.nu_n.array() * (elastic.r_n.array() + elastic.d_lambda.array()) - elastic.r_s_n.array()) /
            denom_n;
        elastic.d_p =
            elastic.d_lambda.array() + lambda_reg * elastic.d_nu_p.array() - elastic.r_p.array();
        elastic.d_n =
            -elastic.d_lambda.array() + lambda_reg * elastic.d_nu_n.array() - elastic.r_n.array();
    }
}

inline void compute_local_model(const vector_const_ref &g,
                                detail::ineq_local_state &ineq,
                                scalar_t rho,
                                scalar_t mu_bar,
                                const vector *mu_t_target,
                                const vector *mu_p_target,
                                const vector *mu_n_target,
                                scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model(ineq) requires lambda_reg >= 0");
    }
    if (g.size() != ineq.t.size() ||
        g.size() != ineq.nu_t.size() ||
        g.size() != ineq.p.size() ||
        g.size() != ineq.n.size()) {
        throw std::runtime_error("compute_local_model(ineq) size mismatch");
    }

    ineq.g_current = g;
    ineq.r_d = g + ineq.t - ineq.p + ineq.n;
    ineq.r_p = vector::Constant(g.size(), rho) - ineq.nu_t - ineq.nu_p;
    ineq.r_n = vector::Constant(g.size(), rho) + ineq.nu_t - ineq.nu_n;
    if (mu_t_target != nullptr && mu_t_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_t_target size mismatch");
    }
    if (mu_p_target != nullptr && mu_p_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_n_target size mismatch");
    }
    for (Eigen::Index i = 0; i < g.size(); ++i) {
        const scalar_t mu_t = mu_t_target ? (*mu_t_target)(i) : mu_bar;
        const scalar_t mu_p = mu_p_target ? (*mu_p_target)(i) : mu_bar;
        const scalar_t mu_n = mu_n_target ? (*mu_n_target)(i) : mu_bar;
        ineq.r_s_t(i) = ineq.nu_t(i) * ineq.t(i) - mu_t;
        ineq.r_s_p(i) = ineq.nu_p(i) * ineq.p(i) - mu_p;
        ineq.r_s_n(i) = ineq.nu_n(i) * ineq.n(i) - mu_n;
    }

    ineq.denom_t = ineq.nu_t.array().max(eps);
    ineq.denom_p = (ineq.p.array() + lambda_reg * ineq.nu_p.array()).max(eps);
    ineq.denom_n = (ineq.n.array() + lambda_reg * ineq.nu_n.array()).max(eps);

    ineq.combo_t = ineq.r_s_t.array() / ineq.denom_t.array();
    ineq.combo_p =
        (ineq.p.array() * ineq.r_p.array() + lambda_reg * ineq.r_s_p.array()) /
        ineq.denom_p.array();
    ineq.combo_n =
        (ineq.n.array() * ineq.r_n.array() + lambda_reg * ineq.r_s_n.array()) /
        ineq.denom_n.array();

    ineq.minv_diag =
        (ineq.t.array() / ineq.denom_t.array() +
         ineq.p.array() / ineq.denom_p.array() +
         ineq.n.array() / ineq.denom_n.array()).inverse();
    ineq.b_d = ineq.r_d - ineq.combo_t + ineq.combo_p - ineq.combo_n;
    ineq.minv_bd = ineq.minv_diag.array() * ineq.b_d.array();
}

inline void compute_local_model(const vector_const_ref &g,
                                detail::ineq_local_state &ineq,
                                scalar_t rho,
                                scalar_t mu_bar,
                                scalar_t lambda_reg) {
    compute_local_model(g, ineq, rho, mu_bar, nullptr, nullptr, nullptr, lambda_reg);
}

inline void recover_local_step(const vector_const_ref &delta_g,
                               detail::ineq_local_state &ineq,
                               scalar_t lambda_reg) {
    if (delta_g.size() != ineq.dim()) {
        throw std::runtime_error("recover_local_step(ineq) size mismatch");
    }
    if (lambda_reg < 0.) {
        throw std::runtime_error("recover_local_step(ineq) requires lambda_reg >= 0");
    }
    ineq.d_nu_t = ineq.minv_diag.array() * (delta_g.array() + ineq.b_d.array());
    ineq.d_nu_p =
        (ineq.nu_p.array() * (ineq.r_p.array() - ineq.d_nu_t.array()) - ineq.r_s_p.array()) /
        ineq.denom_p.array();
    ineq.d_nu_n =
        (ineq.nu_n.array() * (ineq.r_n.array() + ineq.d_nu_t.array()) - ineq.r_s_n.array()) /
        ineq.denom_n.array();
    ineq.d_t =
        -(ineq.r_s_t.array() + ineq.t.array() * ineq.d_nu_t.array()) /
        ineq.denom_t.array();
    ineq.d_p =
        ineq.d_nu_t.array() + lambda_reg * ineq.d_nu_p.array() - ineq.r_p.array();
    ineq.d_n =
        -ineq.d_nu_t.array() + lambda_reg * ineq.d_nu_n.array() - ineq.r_n.array();
}

inline local_residual_summary current_local_residuals(const detail::eq_local_state &elastic) {
    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }
    out.inf_prim = elastic.r_c.cwiseAbs().maxCoeff();
    out.inf_stat = std::max(elastic.r_p.cwiseAbs().maxCoeff(), elastic.r_n.cwiseAbs().maxCoeff());
    out.inf_comp = std::max(elastic.r_s_p.cwiseAbs().maxCoeff(), elastic.r_s_n.cwiseAbs().maxCoeff());
    return out;
}

inline local_residual_summary linearized_newton_residuals(const vector_const_ref &delta_c,
                                                          const detail::eq_local_state &elastic,
                                                          scalar_t lambda_reg) {
    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }
    const vector res_c = delta_c - elastic.d_p + elastic.d_n + elastic.r_c;
    const vector res_p = elastic.d_p - elastic.d_lambda - lambda_reg * elastic.d_nu_p + elastic.r_p;
    const vector res_n = elastic.d_n + elastic.d_lambda - lambda_reg * elastic.d_nu_n + elastic.r_n;
    const vector res_sp =
        elastic.nu_p.cwiseProduct(elastic.d_p) + elastic.p.cwiseProduct(elastic.d_nu_p) + elastic.r_s_p;
    const vector res_sn =
        elastic.nu_n.cwiseProduct(elastic.d_n) + elastic.n.cwiseProduct(elastic.d_nu_n) + elastic.r_s_n;
    out.inf_prim = res_c.cwiseAbs().maxCoeff();
    out.inf_stat = std::max(res_p.cwiseAbs().maxCoeff(), res_n.cwiseAbs().maxCoeff());
    out.inf_comp = std::max(res_sp.cwiseAbs().maxCoeff(), res_sn.cwiseAbs().maxCoeff());
    return out;
}

inline local_residual_summary current_local_residuals(const detail::ineq_local_state &ineq) {
    local_residual_summary out;
    if (ineq.dim() == 0) {
        return out;
    }
    out.inf_prim = ineq.r_d.cwiseAbs().maxCoeff();
    out.inf_stat = std::max({ineq.r_p.cwiseAbs().maxCoeff(),
                             ineq.r_n.cwiseAbs().maxCoeff()});
    out.inf_comp = std::max({ineq.r_s_t.cwiseAbs().maxCoeff(),
                             ineq.r_s_p.cwiseAbs().maxCoeff(),
                             ineq.r_s_n.cwiseAbs().maxCoeff()});
    return out;
}

inline local_residual_summary linearized_newton_residuals(const vector_const_ref &delta_g,
                                                          const detail::ineq_local_state &ineq,
                                                          scalar_t lambda_reg) {
    local_residual_summary out;
    if (ineq.dim() == 0) {
        return out;
    }
    const vector res_d = delta_g + ineq.d_t - ineq.d_p + ineq.d_n + ineq.r_d;
    const vector res_p = ineq.d_p - ineq.d_nu_t - lambda_reg * ineq.d_nu_p + ineq.r_p;
    const vector res_n = ineq.d_n + ineq.d_nu_t - lambda_reg * ineq.d_nu_n + ineq.r_n;
    const vector res_st =
        ineq.nu_t.cwiseProduct(ineq.d_t) + ineq.t.cwiseProduct(ineq.d_nu_t) + ineq.r_s_t;
    const vector res_sp =
        ineq.nu_p.cwiseProduct(ineq.d_p) + ineq.p.cwiseProduct(ineq.d_nu_p) + ineq.r_s_p;
    const vector res_sn =
        ineq.nu_n.cwiseProduct(ineq.d_n) + ineq.n.cwiseProduct(ineq.d_nu_n) + ineq.r_s_n;
    out.inf_prim = res_d.cwiseAbs().maxCoeff();
    out.inf_stat = std::max({res_p.cwiseAbs().maxCoeff(), res_n.cwiseAbs().maxCoeff()});
    out.inf_comp = std::max({res_st.cwiseAbs().maxCoeff(),
                             res_sp.cwiseAbs().maxCoeff(),
                             res_sn.cwiseAbs().maxCoeff()});
    return out;
}

} // namespace moto::solver::restoration
