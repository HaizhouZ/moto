#pragma once

#include <moto/core/fwd.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>

#include <stdexcept>

namespace moto::solver::restoration {

/**
 * @brief IPOPT-style elastic restoration initialization for one scalar residual.
 *
 * @details For fixed residual c, initialize p,n > 0 from the barrier-smoothed
 * exact-penalty subproblem
 *   min rho (p + n) - mu log p - mu log n
 *   s.t. c - p + n = 0.
 * Then initialize the associated bound duals by z_p = mu / p, z_n = mu / n and
 * the elastic equality multiplier by lambda = rho - z_p = z_n - rho.
 */
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

struct local_residual_summary {
    scalar_t inf_c = 0.;
    scalar_t inf_p = 0.;
    scalar_t inf_n = 0.;
    scalar_t inf_s_p = 0.;
    scalar_t inf_s_n = 0.;

    scalar_t inf_stationarity() const { return std::max(inf_p, inf_n); }
    scalar_t inf_complementarity() const { return std::max(inf_s_p, inf_s_n); }
    scalar_t inf_all() const {
        return std::max({inf_c, inf_p, inf_n, inf_s_p, inf_s_n});
    }
};

inline void compute_local_model(const vector_const_ref &c,
                                const vector_const_ref &lambda,
                                resto_elastic_constr &elastic,
                                scalar_t rho,
                                scalar_t mu_bar,
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

    elastic.r_s_p = elastic.nu_p.array() * elastic.p.array() - mu_bar;
    elastic.r_s_n = elastic.nu_n.array() * elastic.n.array() - mu_bar;

    if (lambda_reg == 0.) {
        const auto denom_p = elastic.nu_p.array().max(eps);
        const auto denom_n = elastic.nu_n.array().max(eps);

        elastic.combo_p =
            (elastic.r_s_p.array() + elastic.p.array() * elastic.r_p.array()) /
            denom_p;
        elastic.combo_n =
            (elastic.r_s_n.array() + elastic.n.array() * elastic.r_n.array()) /
            denom_n;

        elastic.minv_diag =
            (elastic.p.array() / denom_p +
             elastic.n.array() / denom_n)
                .inverse();
    } else {
        const auto denom_p = elastic.p.array() + lambda_reg * elastic.nu_p.array();
        const auto denom_n = elastic.n.array() + lambda_reg * elastic.nu_n.array();

        elastic.combo_p =
            (elastic.p.array() * elastic.r_p.array() +
             lambda_reg * elastic.r_s_p.array()) /
            denom_p.max(eps);
        elastic.combo_n =
            (elastic.n.array() * elastic.r_n.array() +
             lambda_reg * elastic.r_s_n.array()) /
            denom_n.max(eps);

        elastic.minv_diag =
            (elastic.p.array() / denom_p.max(eps) +
             elastic.n.array() / denom_n.max(eps))
                .inverse();
    }

    elastic.b_c =
        elastic.r_c + elastic.combo_p - elastic.combo_n;
    elastic.minv_bc = elastic.minv_diag.array() * elastic.b_c.array();
}

inline void recover_local_step(const vector_const_ref &delta_c,
                               resto_elastic_constr &elastic,
                               scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (lambda_reg < 0.) {
        throw std::runtime_error("recover_local_step requires lambda_reg >= 0");
    }
    if (delta_c.size() != elastic.dim()) {
        throw std::runtime_error("recover_local_step size mismatch");
    }

    elastic.d_lambda =
        elastic.minv_diag.array() * (delta_c.array() + elastic.b_c.array());

    if (lambda_reg == 0.) {
        elastic.d_nu_p = elastic.r_p - elastic.d_lambda;
        elastic.d_nu_n = elastic.r_n + elastic.d_lambda;

        elastic.d_p =
            -(elastic.r_s_p.array() +
              elastic.p.array() * elastic.d_nu_p.array()) /
            elastic.nu_p.array().max(eps);
        elastic.d_n =
            -(elastic.r_s_n.array() +
              elastic.n.array() * elastic.d_nu_n.array()) /
            elastic.nu_n.array().max(eps);
    } else {
        const auto denom_p = (elastic.p.array() + lambda_reg * elastic.nu_p.array()).max(eps);
        const auto denom_n = (elastic.n.array() + lambda_reg * elastic.nu_n.array()).max(eps);

        elastic.d_nu_p =
            (elastic.nu_p.array() * (elastic.r_p.array() - elastic.d_lambda.array()) -
             elastic.r_s_p.array()) /
            denom_p;
        elastic.d_nu_n =
            (elastic.nu_n.array() * (elastic.r_n.array() + elastic.d_lambda.array()) -
             elastic.r_s_n.array()) /
            denom_n;

        elastic.d_p =
            elastic.d_lambda.array() + lambda_reg * elastic.d_nu_p.array() -
            elastic.r_p.array();
        elastic.d_n =
            -elastic.d_lambda.array() + lambda_reg * elastic.d_nu_n.array() -
            elastic.r_n.array();
    }
}

inline local_residual_summary current_local_residuals(const resto_elastic_constr &elastic) {
    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }
    out.inf_c = elastic.r_c.cwiseAbs().maxCoeff();
    out.inf_p = elastic.r_p.cwiseAbs().maxCoeff();
    out.inf_n = elastic.r_n.cwiseAbs().maxCoeff();
    out.inf_s_p = elastic.r_s_p.cwiseAbs().maxCoeff();
    out.inf_s_n = elastic.r_s_n.cwiseAbs().maxCoeff();
    return out;
}

inline local_residual_summary recovered_linearized_residuals(const vector_const_ref &delta_c,
                                                             const resto_elastic_constr &elastic,
                                                             scalar_t lambda_reg) {
    if (delta_c.size() != elastic.dim()) {
        throw std::runtime_error("recovered_linearized_residuals size mismatch");
    }

    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }

    const auto res_c =
        delta_c.array() - elastic.d_p.array() + elastic.d_n.array() + elastic.r_c.array();
    const auto res_p =
        elastic.d_p.array() - elastic.d_lambda.array() -
        lambda_reg * elastic.d_nu_p.array() + elastic.r_p.array();
    const auto res_n =
        elastic.d_n.array() + elastic.d_lambda.array() -
        lambda_reg * elastic.d_nu_n.array() + elastic.r_n.array();
    const auto res_s_p =
        elastic.nu_p.array() * elastic.d_p.array() +
        elastic.p.array() * elastic.d_nu_p.array() + elastic.r_s_p.array();
    const auto res_s_n =
        elastic.nu_n.array() * elastic.d_n.array() +
        elastic.n.array() * elastic.d_nu_n.array() + elastic.r_s_n.array();

    out.inf_c = res_c.abs().maxCoeff();
    out.inf_p = res_p.abs().maxCoeff();
    out.inf_n = res_n.abs().maxCoeff();
    out.inf_s_p = res_s_p.abs().maxCoeff();
    out.inf_s_n = res_s_n.abs().maxCoeff();
    return out;
}

inline local_residual_summary predicted_trial_residuals(const vector_const_ref &delta_c,
                                                        const vector_const_ref &lambda,
                                                        const resto_elastic_constr &elastic,
                                                        scalar_t rho,
                                                        scalar_t mu_bar) {
    if (delta_c.size() != elastic.dim() || lambda.size() != elastic.dim()) {
        throw std::runtime_error("predicted_trial_residuals size mismatch");
    }

    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }

    const auto c_trial = elastic.c_current.array() + delta_c.array();
    const auto p_trial = elastic.p.array() + elastic.d_p.array();
    const auto n_trial = elastic.n.array() + elastic.d_n.array();
    const auto nu_p_trial = elastic.nu_p.array() + elastic.d_nu_p.array();
    const auto nu_n_trial = elastic.nu_n.array() + elastic.d_nu_n.array();
    const auto lambda_trial = lambda.array() + elastic.d_lambda.array();

    const auto res_c = c_trial - p_trial + n_trial;
    const auto res_p = scalar_t(rho) - lambda_trial - nu_p_trial;
    const auto res_n = scalar_t(rho) + lambda_trial - nu_n_trial;
    const auto res_s_p = nu_p_trial * p_trial - scalar_t(mu_bar);
    const auto res_s_n = nu_n_trial * n_trial - scalar_t(mu_bar);

    out.inf_c = res_c.abs().maxCoeff();
    out.inf_p = res_p.abs().maxCoeff();
    out.inf_n = res_n.abs().maxCoeff();
    out.inf_s_p = res_s_p.abs().maxCoeff();
    out.inf_s_n = res_s_n.abs().maxCoeff();
    return out;
}

} // namespace moto::solver::restoration
