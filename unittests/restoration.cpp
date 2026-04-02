#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>
#include <moto/solver/restoration/resto_runtime.hpp>

using namespace moto;
using namespace moto::solver;
using namespace moto::solver::restoration;

namespace {
bool approx_zero(const vector &v, scalar_t tol = 1e-9) {
    return v.size() == 0 || v.cwiseAbs().maxCoeff() < tol;
}

bool approx_scalar(scalar_t a, scalar_t b, scalar_t tol = 1e-12) {
    return std::abs(a - b) < tol;
}
} // namespace

TEST_CASE("restoration equality local KKT recovery satisfies regularized linearization") {
    resto_elastic_constr eq;
    eq.resize(1, 2);
    eq.p << 0.7, 0.9, 1.2;
    eq.n << 0.8, 1.1, 0.6;
    eq.nu_p << 1.3, 0.7, 0.9;
    eq.nu_n << 0.6, 1.2, 1.1;

    vector c(3);
    c << 0.2, -0.4, 0.3;
    vector lambda(3);
    lambda << 0.1, -0.2, 0.15;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;
    const scalar_t lambda_reg = 1e-3;

    compute_local_model(c, lambda, eq, rho, mu_bar, lambda_reg);

    vector delta_c(3);
    delta_c << -0.11, 0.07, 0.19;
    recover_local_step(delta_c, eq, lambda_reg);

    const vector res_c = delta_c - eq.d_p + eq.d_n + eq.r_c;
    const vector res_p = eq.d_p - eq.d_lambda - lambda_reg * eq.d_nu_p + eq.r_p;
    const vector res_n = eq.d_n + eq.d_lambda - lambda_reg * eq.d_nu_n + eq.r_n;
    const vector res_sp = eq.nu_p.cwiseProduct(eq.d_p) + eq.p.cwiseProduct(eq.d_nu_p) + eq.r_s_p;
    const vector res_sn = eq.nu_n.cwiseProduct(eq.d_n) + eq.n.cwiseProduct(eq.d_nu_n) + eq.r_s_n;

    REQUIRE(approx_zero(res_c));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
}

TEST_CASE("restoration inequality local KKT recovery satisfies regularized linearization") {
    resto_ineq_constr iq;
    iq.resize(2, 1);
    iq.t << 1.4, 1.1, 0.9;
    iq.p << 0.8, 1.0, 1.3;
    iq.n << 0.9, 0.7, 1.2;
    iq.nu_t << 0.5, 0.6, 0.8;
    iq.nu_p << 0.9, 0.8, 0.7;
    iq.nu_n << 0.6, 0.9, 0.5;

    vector g(3);
    g << -0.2, 0.5, -0.1;
    vector lambda(3);
    lambda << 0.15, -0.05, 0.2;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;
    const scalar_t lambda_reg = 1e-3;

    compute_local_model(g, lambda, iq, rho, mu_bar, lambda_reg);

    vector delta_g(3);
    delta_g << 0.08, -0.12, 0.04;
    recover_local_step(delta_g, iq, lambda_reg);

    const vector res_d = delta_g + iq.d_t - iq.d_p + iq.d_n + iq.r_d;
    const vector res_t = iq.d_t + iq.d_lambda - lambda_reg * iq.d_nu_t + iq.r_t;
    const vector res_p = iq.d_p - iq.d_lambda - lambda_reg * iq.d_nu_p + iq.r_p;
    const vector res_n = iq.d_n + iq.d_lambda - lambda_reg * iq.d_nu_n + iq.r_n;
    const vector res_st = iq.nu_t.cwiseProduct(iq.d_t) + iq.t.cwiseProduct(iq.d_nu_t) + iq.r_s_t;
    const vector res_sp = iq.nu_p.cwiseProduct(iq.d_p) + iq.p.cwiseProduct(iq.d_nu_p) + iq.r_s_p;
    const vector res_sn = iq.nu_n.cwiseProduct(iq.d_n) + iq.n.cwiseProduct(iq.d_nu_n) + iq.r_s_n;

    REQUIRE(approx_zero(res_d));
    REQUIRE(approx_zero(res_t));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_st));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
}

TEST_CASE("positivity helper reuses consistent alpha and backup semantics") {
    vector primal(3), primal_step(3), primal_backup(3);
    primal << 1.0, 2.0, 0.5;
    primal_step << -0.2, 0.1, -0.4;

    vector dual(3), dual_step(3), dual_backup(3);
    dual << 0.8, 1.5, 0.9;
    dual_step << -0.1, 0.2, -0.3;

    solver::linesearch_config cfg;
    positivity::update_pair_bounds(cfg, primal, primal_step, dual, dual_step);
    REQUIRE(approx_scalar(cfg.primal.alpha_max, std::min(1.0, -0.995 * 0.5 / -0.4)));
    REQUIRE(approx_scalar(cfg.dual.alpha_max, std::min(1.0, -0.995 * 0.9 / -0.3)));

    positivity::backup_pair(primal, primal_backup, dual, dual_backup);
    positivity::apply_pair_step(primal, primal_step, 0.5, dual, dual_step, 0.25);
    REQUIRE(approx_scalar(primal(0), 0.9));
    REQUIRE(approx_scalar(dual(2), 0.825));

    positivity::restore_pair(primal, primal_backup, dual, dual_backup);
    REQUIRE(approx_scalar(primal(0), 1.0));
    REQUIRE(approx_scalar(dual(2), 0.9));
}

TEST_CASE("restoration multiplier reset helper follows threshold policy") {
    vector multiplier(3);
    multiplier << 1.0, -5.0, 2.0;

    REQUIRE_FALSE(should_reset_multiplier(multiplier, 10.0));
    REQUIRE(should_reset_multiplier(multiplier, 4.0));
    REQUIRE(should_reset_multiplier(multiplier, 0.0));

    maybe_reset_multiplier(multiplier, 4.0, 1.0);
    REQUIRE(approx_scalar(multiplier(0), 1.0));
    REQUIRE(approx_scalar(multiplier(1), 1.0));
    REQUIRE(approx_scalar(multiplier(2), 1.0));
}
