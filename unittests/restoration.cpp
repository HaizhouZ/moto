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

TEST_CASE("restoration exit helper restores outer dual arrays") {
    array_type<vector, constr_fields> dual;
    array_type<vector, constr_fields> backup;

    scalar_t seed = 1.0;
    for (auto f : constr_fields) {
        dual[f] = vector::Constant(2, seed);
        backup[f] = vector::Constant(2, 10.0 + seed);
        seed += 1.0;
    }

    restore_outer_duals(dual, backup);

    for (auto f : constr_fields) {
        REQUIRE(dual[f].isApprox(backup[f]));
    }
}

TEST_CASE("restoration exit helper copies slack and resets bound multiplier when threshold exceeded") {
    vector slack(3), multiplier(3), resto_slack(3), resto_multiplier(3);
    slack << 1.0, 1.0, 1.0;
    multiplier << 2.0, 2.0, 2.0;
    resto_slack << 0.6, 0.7, 0.8;
    resto_multiplier << 5.0, -3.0, 2.0;

    commit_bound_state(slack, multiplier, resto_slack, resto_multiplier, 4.0, 1.0);

    REQUIRE(slack.isApprox(resto_slack));
    REQUIRE(approx_scalar(multiplier(0), 1.0));
    REQUIRE(approx_scalar(multiplier(1), 1.0));
    REQUIRE(approx_scalar(multiplier(2), 1.0));
}

TEST_CASE("restoration exit helper preserves bound multiplier when below threshold") {
    vector slack(2), multiplier(2), resto_slack(2), resto_multiplier(2);
    slack << 1.0, 1.0;
    multiplier << 2.0, 2.0;
    resto_slack << 0.6, 0.7;
    resto_multiplier << 3.0, -2.0;

    commit_bound_state(slack, multiplier, resto_slack, resto_multiplier, 4.0, 1.0);

    REQUIRE(slack.isApprox(resto_slack));
    REQUIRE(multiplier.isApprox(resto_multiplier));
}

TEST_CASE("restoration equality dual reset helper only clears equality fields") {
    array_type<vector, constr_fields> dual;
    dual[__dyn] = vector::Constant(2, 7.0);
    dual[__eq_x] = vector::Constant(2, 3.0);
    dual[__eq_xu] = vector::Constant(1, -5.0);
    dual[__ineq_x] = vector::Constant(2, 11.0);
    dual[__ineq_xu] = vector::Constant(1, -13.0);

    reset_equality_duals(dual, 4.0);

    REQUIRE(dual[__dyn].isConstant(7.0));
    REQUIRE(dual[__eq_x].isZero());
    REQUIRE(dual[__eq_xu].isZero());
    REQUIRE(dual[__ineq_x].isConstant(11.0));
    REQUIRE(dual[__ineq_xu].isConstant(-13.0));
}

TEST_CASE("restoration local residual helper reports local stationarity and complementarity maxima") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.elastic_eq.resize(1, 1);
    aux.elastic_eq.r_p << 0.2, -0.6;
    aux.elastic_eq.r_n << -0.3, 0.1;
    aux.elastic_eq.r_s_p << 0.4, -0.5;
    aux.elastic_eq.r_s_n << 0.05, -0.2;

    aux.elastic_ineq.resize(1, 1);
    aux.elastic_ineq.r_t << -0.7, 0.25;
    aux.elastic_ineq.r_p << 0.15, -0.35;
    aux.elastic_ineq.r_n << -0.1, 0.05;
    aux.elastic_ineq.r_s_t << 0.9, -0.2;
    aux.elastic_ineq.r_s_p << -0.45, 0.3;
    aux.elastic_ineq.r_s_n << 0.12, -0.18;

    const auto info = refinement_local_residuals(aux);
    REQUIRE(approx_scalar(info.stationarity, 0.7));
    REQUIRE(approx_scalar(info.complementarity, 0.9));
}

TEST_CASE("restoration reduced residual combines active w stationarity with local residual blocks") {
    array_type<row_vector, primal_fields> lag_jac;
    array_type<row_vector, primal_fields> lag_jac_corr;
    for (auto pf : primal_fields) {
        lag_jac[pf].resize(2);
        lag_jac_corr[pf].resize(2);
        lag_jac[pf].setZero();
        lag_jac_corr[pf].setZero();
    }
    lag_jac[__x] << 1.0, -2.0;
    lag_jac[__u] << 0.5, -0.2;
    lag_jac[__y] << -0.7, 0.3;
    lag_jac_corr[__x] << -0.1, 0.4;
    lag_jac_corr[__u] << 0.2, 0.1;
    lag_jac_corr[__y] << -0.3, 0.2;

    vector dyn_res(2);
    dyn_res << 0.4, -1.2;

    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.r_c << -0.6;
    aux.elastic_eq.r_p << 0.25;
    aux.elastic_eq.r_n << -0.15;
    aux.elastic_eq.r_s_p << 0.5;
    aux.elastic_eq.r_s_n << -0.35;

    aux.elastic_ineq.resize(0, 1);
    aux.elastic_ineq.r_d << 0.8;
    aux.elastic_ineq.r_t << -0.9;
    aux.elastic_ineq.r_p << 0.45;
    aux.elastic_ineq.r_n << -0.2;
    aux.elastic_ineq.r_s_t << 0.7;
    aux.elastic_ineq.r_s_p << -0.25;
    aux.elastic_ineq.r_s_n << 0.1;

    const auto residual = compute_reduced_residual(lag_jac, lag_jac_corr, dyn_res, aux);
    REQUIRE(residual.w_stationarity[__x].isApprox((row_vector(2) << 0.9, -1.6).finished()));
    REQUIRE(residual.w_stationarity[__u].isApprox((row_vector(2) << 0.7, -0.1).finished()));
    REQUIRE(residual.w_stationarity[__y].isApprox((row_vector(2) << -1.0, 0.5).finished()));
    REQUIRE(approx_scalar(residual.eq_local.inf_prim, 0.6));
    REQUIRE(approx_scalar(residual.eq_local.inf_stat, 0.25));
    REQUIRE(approx_scalar(residual.eq_local.inf_comp, 0.5));
    REQUIRE(approx_scalar(residual.ineq_local.inf_prim, 0.8));
    REQUIRE(approx_scalar(residual.ineq_local.inf_stat, 0.9));
    REQUIRE(approx_scalar(residual.ineq_local.inf_comp, 0.7));
    REQUIRE(approx_scalar(residual.inf_primal, 1.2));
    REQUIRE(approx_scalar(residual.inf_dual, 1.6));
    REQUIRE(approx_scalar(residual.inf_comp, 0.7));
}

TEST_CASE("restoration barrier stats average local complementarity products") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.p << 2.0;
    aux.elastic_eq.nu_p << 3.0;
    aux.elastic_eq.n << 5.0;
    aux.elastic_eq.nu_n << 7.0;

    aux.elastic_ineq.resize(0, 1);
    aux.elastic_ineq.t << 11.0;
    aux.elastic_ineq.nu_t << 13.0;
    aux.elastic_ineq.p << 17.0;
    aux.elastic_ineq.nu_p << 19.0;
    aux.elastic_ineq.n << 23.0;
    aux.elastic_ineq.nu_n << 29.0;

    const auto stats = current_barrier_stats(aux);
    REQUIRE(stats.n_comp == 5);
    REQUIRE(approx_scalar(stats.avg_comp, (6.0 + 35.0 + 143.0 + 323.0 + 667.0) / 5.0));
    REQUIRE(approx_scalar(stats.inf_comp, 667.0));
}

TEST_CASE("restoration mu update follows local complementarity average in adaptive mode") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.mu_bar = 10.0;
    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.p << 2.0;
    aux.elastic_eq.nu_p << 1.0;
    aux.elastic_eq.n << 3.0;
    aux.elastic_eq.nu_n << 2.0;

    solver::ipm_config cfg;
    cfg.mu_method = solver::ipm_config::mehrotra_predictor_corrector;

    REQUIRE(update_mu_bar(aux, cfg, 10.0, 0.2, 1.0, 1.0));
    REQUIRE(approx_scalar(aux.mu_bar, 4.0));
}

TEST_CASE("restoration mu update follows monotone decrease threshold") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.mu_bar = 2.0;
    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.p << 1.0;
    aux.elastic_eq.nu_p << 2.0;
    aux.elastic_eq.n << 1.0;
    aux.elastic_eq.nu_n << 2.0;

    solver::ipm_config cfg;
    cfg.mu_method = solver::ipm_config::monotonic_decrease;
    const scalar_t mu_monotone_fraction_threshold = 10.0;
    const scalar_t mu_monotone_factor = 0.2;

    REQUIRE(update_mu_bar(aux, cfg, mu_monotone_fraction_threshold, mu_monotone_factor, 1.0, 1.0));
    REQUIRE(aux.mu_bar < 2.0);
    REQUIRE(aux.mu_bar >= 1e-11);
}

TEST_CASE("restoration correction rhs only loads reduced u and y stationarity") {
    array_type<row_vector, primal_fields> rhs;
    for (auto pf : primal_fields) {
        rhs[pf].resize(2);
        rhs[pf].setConstant(5.0);
    }

    reduced_residual_info residual;
    residual.w_stationarity[__x].resize(2);
    residual.w_stationarity[__u].resize(2);
    residual.w_stationarity[__y].resize(2);
    residual.w_stationarity[__x] << 1.0, 2.0;
    residual.w_stationarity[__u] << -0.4, 0.6;
    residual.w_stationarity[__y] << 0.3, -0.9;

    load_correction_rhs(rhs, residual);

    REQUIRE(rhs[__x].isZero());
    REQUIRE(rhs[__u].isApprox(residual.w_stationarity[__u]));
    REQUIRE(rhs[__y].isApprox(residual.w_stationarity[__y]));
}
