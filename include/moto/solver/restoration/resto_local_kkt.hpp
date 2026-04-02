#pragma once

#include <moto/core/fwd.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>

#include <stdexcept>

namespace moto::solver::restoration {

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
    scalar_t lambda = 0.;
};

inline elastic_init_ineq_scalar initialize_elastic_ineq_scalar(scalar_t g, scalar_t rho, scalar_t mu_bar) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires mu_bar > 0");
    }

    const scalar_t eps = std::max(scalar_t(1e-16), std::min(scalar_t(1e-12), scalar_t(0.25) * rho));
    const scalar_t lambda_hi = rho - eps;
    if (!(lambda_hi > eps)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires rho sufficiently larger than epsilon");
    }

    const auto residual = [&](scalar_t lambda) {
        return g + mu_bar / lambda - mu_bar / (rho - lambda) + mu_bar / (rho + lambda);
    };

    scalar_t lo = eps;
    scalar_t hi = lambda_hi;
    scalar_t f_lo = residual(lo);
    scalar_t f_hi = residual(hi);
    if (!(f_lo > 0.) || !(f_hi < 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar failed to bracket local KKT root");
    }

    for (int iter = 0; iter < 80; ++iter) {
        const scalar_t mid = scalar_t(0.5) * (lo + hi);
        const scalar_t f_mid = residual(mid);
        if (f_mid > 0.) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    elastic_init_ineq_scalar out;
    out.lambda = scalar_t(0.5) * (lo + hi);
    out.nu_t = out.lambda;
    out.nu_p = rho - out.lambda;
    out.nu_n = rho + out.lambda;
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

inline void recover_local_step(const vector_const_ref &delta_c,
                               resto_elastic_constr &elastic,
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
                                const vector_const_ref &lambda,
                                resto_ineq_constr &ineq,
                                scalar_t rho,
                                scalar_t mu_bar,
                                scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model(ineq) requires lambda_reg >= 0");
    }
    if (g.size() != lambda.size() ||
        g.size() != ineq.t.size() ||
        g.size() != ineq.p.size() ||
        g.size() != ineq.n.size()) {
        throw std::runtime_error("compute_local_model(ineq) size mismatch");
    }

    ineq.g_current = g;
    ineq.r_d = g + ineq.t - ineq.p + ineq.n;
    ineq.r_t = lambda - ineq.nu_t;
    ineq.r_p = vector::Constant(g.size(), rho) - lambda - ineq.nu_p;
    ineq.r_n = vector::Constant(g.size(), rho) + lambda - ineq.nu_n;
    ineq.r_s_t = ineq.nu_t.array() * ineq.t.array() - mu_bar;
    ineq.r_s_p = ineq.nu_p.array() * ineq.p.array() - mu_bar;
    ineq.r_s_n = ineq.nu_n.array() * ineq.n.array() - mu_bar;

    if (lambda_reg == 0.) {
        const auto denom_t = ineq.nu_t.array().max(eps);
        const auto denom_p = ineq.nu_p.array().max(eps);
        const auto denom_n = ineq.nu_n.array().max(eps);
        ineq.combo_t =
            (ineq.r_s_t.array() + ineq.t.array() * ineq.r_t.array()) / denom_t;
        ineq.combo_p =
            (ineq.r_s_p.array() + ineq.p.array() * ineq.r_p.array()) / denom_p;
        ineq.combo_n =
            (ineq.r_s_n.array() + ineq.n.array() * ineq.r_n.array()) / denom_n;
        ineq.minv_diag =
            (ineq.t.array() / denom_t +
             ineq.p.array() / denom_p +
             ineq.n.array() / denom_n)
                .inverse();
    } else {
        const auto denom_t = (ineq.t.array() + lambda_reg * ineq.nu_t.array()).max(eps);
        const auto denom_p = (ineq.p.array() + lambda_reg * ineq.nu_p.array()).max(eps);
        const auto denom_n = (ineq.n.array() + lambda_reg * ineq.nu_n.array()).max(eps);
        ineq.combo_t =
            (ineq.t.array() * ineq.r_t.array() + lambda_reg * ineq.r_s_t.array()) / denom_t;
        ineq.combo_p =
            (ineq.p.array() * ineq.r_p.array() + lambda_reg * ineq.r_s_p.array()) / denom_p;
        ineq.combo_n =
            (ineq.n.array() * ineq.r_n.array() + lambda_reg * ineq.r_s_n.array()) / denom_n;
        ineq.minv_diag =
            (ineq.t.array() / denom_t +
             ineq.p.array() / denom_p +
             ineq.n.array() / denom_n)
                .inverse();
    }

    ineq.b_d = ineq.r_d - ineq.combo_t + ineq.combo_p - ineq.combo_n;
    ineq.minv_bd = ineq.minv_diag.array() * ineq.b_d.array();
}

inline void recover_local_step(const vector_const_ref &delta_g,
                               resto_ineq_constr &ineq,
                               scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (delta_g.size() != ineq.dim()) {
        throw std::runtime_error("recover_local_step(ineq) size mismatch");
    }
    ineq.d_lambda = ineq.minv_diag.array() * (delta_g.array() + ineq.b_d.array());
    if (lambda_reg == 0.) {
        ineq.d_nu_t = ineq.r_t + ineq.d_lambda;
        ineq.d_nu_p = ineq.r_p - ineq.d_lambda;
        ineq.d_nu_n = ineq.r_n + ineq.d_lambda;
        ineq.d_t =
            -(ineq.r_s_t.array() + ineq.t.array() * ineq.d_nu_t.array()) /
            ineq.nu_t.array().max(eps);
        ineq.d_p =
            -(ineq.r_s_p.array() + ineq.p.array() * ineq.d_nu_p.array()) /
            ineq.nu_p.array().max(eps);
        ineq.d_n =
            -(ineq.r_s_n.array() + ineq.n.array() * ineq.d_nu_n.array()) /
            ineq.nu_n.array().max(eps);
    } else {
        const auto denom_t = (ineq.t.array() + lambda_reg * ineq.nu_t.array()).max(eps);
        const auto denom_p = (ineq.p.array() + lambda_reg * ineq.nu_p.array()).max(eps);
        const auto denom_n = (ineq.n.array() + lambda_reg * ineq.nu_n.array()).max(eps);
        ineq.d_nu_t =
            (ineq.nu_t.array() * (ineq.r_t.array() + ineq.d_lambda.array()) - ineq.r_s_t.array()) /
            denom_t;
        ineq.d_nu_p =
            (ineq.nu_p.array() * (ineq.r_p.array() - ineq.d_lambda.array()) - ineq.r_s_p.array()) /
            denom_p;
        ineq.d_nu_n =
            (ineq.nu_n.array() * (ineq.r_n.array() + ineq.d_lambda.array()) - ineq.r_s_n.array()) /
            denom_n;
        ineq.d_t =
            -ineq.d_lambda.array() + lambda_reg * ineq.d_nu_t.array() - ineq.r_t.array();
        ineq.d_p =
            ineq.d_lambda.array() + lambda_reg * ineq.d_nu_p.array() - ineq.r_p.array();
        ineq.d_n =
            -ineq.d_lambda.array() + lambda_reg * ineq.d_nu_n.array() - ineq.r_n.array();
    }
}

inline local_residual_summary current_local_residuals(const resto_elastic_constr &elastic) {
    local_residual_summary out;
    if (elastic.dim() == 0) {
        return out;
    }
    out.inf_prim = elastic.r_c.cwiseAbs().maxCoeff();
    out.inf_stat = std::max(elastic.r_p.cwiseAbs().maxCoeff(), elastic.r_n.cwiseAbs().maxCoeff());
    out.inf_comp = std::max(elastic.r_s_p.cwiseAbs().maxCoeff(), elastic.r_s_n.cwiseAbs().maxCoeff());
    return out;
}

inline local_residual_summary current_local_residuals(const resto_ineq_constr &ineq) {
    local_residual_summary out;
    if (ineq.dim() == 0) {
        return out;
    }
    out.inf_prim = ineq.r_d.cwiseAbs().maxCoeff();
    out.inf_stat = std::max({ineq.r_t.cwiseAbs().maxCoeff(),
                             ineq.r_p.cwiseAbs().maxCoeff(),
                             ineq.r_n.cwiseAbs().maxCoeff()});
    out.inf_comp = std::max({ineq.r_s_t.cwiseAbs().maxCoeff(),
                             ineq.r_s_p.cwiseAbs().maxCoeff(),
                             ineq.r_s_n.cwiseAbs().maxCoeff()});
    return out;
}

} // namespace moto::solver::restoration
