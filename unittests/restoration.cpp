#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/restoration/resto_local_kkt.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>
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

TEST_CASE("restoration overlay problem keeps dyn and replaces non-dynamics with standard overlays") {
    auto prob = ocp::create();

    auto [x, y] = sym::states("x", 2);
    auto u = sym::inputs("u", 1);

    prob->add(*x);
    prob->add(*u);
    prob->add(*y);

    auto cost_stage = cost(new generic_cost("stage_cost", approx_order::second));
    dynamic_cast<generic_func &>(*cost_stage).add_argument(u);
    cost_stage->value = [](func_approx_data &d) {
        d.v_(0) += scalar_t(0.5) * d[0].squaredNorm();
    };
    cost_stage->jacobian = [](func_approx_data &d) {
        d.lag_jac_[0].noalias() += d[0].transpose();
    };
    cost_stage->hessian = [](func_approx_data &d) {
        d.lag_hess_[0][0].diagonal().array() += 1.;
    };
    prob->add(*cost_stage);

    auto eq = constr(new generic_constr("eq", approx_order::second, 1));
    eq->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*eq).add_argument(x);
    eq->value = [](func_approx_data &d) { d.v_(0) = d[0](0); };
    eq->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    prob->add(*eq);

    auto iq = constr(new generic_constr("iq", approx_order::second, 1));
    iq->field_hint().is_eq = false;
    dynamic_cast<generic_func &>(*iq).add_argument(u);
    iq->value = [](func_approx_data &d) { d.v_(0) = d[0](0) - scalar_t(1.0); };
    iq->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    prob->add(*iq);

    auto dyn = constr(new generic_constr("dyn", approx_order::first, 2, __dyn));
    dynamic_cast<generic_func &>(*dyn).add_argument(x);
    dynamic_cast<generic_func &>(*dyn).add_argument(u);
    dynamic_cast<generic_func &>(*dyn).add_argument(y);
    dyn->value = [](func_approx_data &d) {
        d.v_ = d[2] - d[0];
    };
    dyn->jacobian = [](func_approx_data &d) {
        d.jac_[0].setZero();
        d.jac_[2].setZero();
        d.jac_[0].diagonal().array() = -1.;
        d.jac_[2].diagonal().array() = 1.;
    };
    prob->add(*dyn);

    prob->wait_until_ready();

    const auto resto = build_restoration_overlay_problem(
        prob,
        restoration_overlay_settings{
            .rho_u = 1e-4,
            .rho_y = 1e-4,
            .rho_eq = 10.0,
            .rho_ineq = 20.0,
            .lambda_reg = 1e-8,
        });

    REQUIRE(resto->num(__dyn) == 1);
    REQUIRE(resto->num(__cost) == 1);
    REQUIRE(resto->num(__eq_x) == 0);
    REQUIRE(resto->num(__eq_x_soft) == 1);
    REQUIRE(resto->num(__ineq_xu) == 1);

    const auto *prox = dynamic_cast<const resto_prox_cost *>(resto->exprs(__cost).front().get());
    REQUIRE(prox != nullptr);

    const auto *eq_overlay = dynamic_cast<const resto_eq_elastic_constr *>(resto->exprs(__eq_x_soft).front().get());
    REQUIRE(eq_overlay != nullptr);
    REQUIRE(eq_overlay->source()->name() == eq->name());

    const auto *ineq_overlay = dynamic_cast<const resto_ineq_elastic_ipm_constr *>(resto->exprs(__ineq_xu).front().get());
    REQUIRE(ineq_overlay != nullptr);
    REQUIRE(ineq_overlay->source()->name() == iq->name());
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

TEST_CASE("restoration inequality initializer satisfies local KKT at the central point") {
    const scalar_t g = -0.35;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;

    const auto init = initialize_elastic_ineq_scalar(g, rho, mu_bar);

    resto_ineq_constr iq;
    iq.resize(1, 0);
    iq.t << init.t;
    iq.p << init.p;
    iq.n << init.n;
    iq.nu_t << init.nu_t;
    iq.nu_p << init.nu_p;
    iq.nu_n << init.nu_n;

    vector g_vec(1);
    g_vec << g;
    vector lambda(1);
    lambda << init.lambda;

    compute_local_model(g_vec, lambda, iq, rho, mu_bar, 0.0);

    REQUIRE(approx_zero(iq.r_d, 1e-10));
    REQUIRE(approx_zero(iq.r_t, 1e-10));
    REQUIRE(approx_zero(iq.r_p, 1e-10));
    REQUIRE(approx_zero(iq.r_n, 1e-10));
    REQUIRE(approx_zero(iq.r_s_t, 1e-10));
    REQUIRE(approx_zero(iq.r_s_p, 1e-10));
    REQUIRE(approx_zero(iq.r_s_n, 1e-10));
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
    REQUIRE(approx_scalar(info.stationarity, 0.6));
    REQUIRE(approx_scalar(info.complementarity, 0.5));
}

TEST_CASE("restoration elastic blocks own their penalty and barrier bookkeeping") {
    resto_elastic_constr eq;
    eq.resize(1, 1);
    eq.p << 2.0, 3.0;
    eq.n << 5.0, 7.0;
    eq.p_backup << 4.0, 8.0;
    eq.n_backup << 2.0, 10.0;
    eq.d_p << 0.5, -1.0;
    eq.d_n << 1.0, 2.0;

    REQUIRE(approx_scalar(eq.penalty_sum(), 17.0));
    REQUIRE(approx_scalar(eq.penalty_dir_deriv(), 2.5));
    REQUIRE(approx_scalar(eq.barrier_log_sum(), std::log(2.0) + std::log(3.0) + std::log(5.0) + std::log(7.0)));
    REQUIRE(approx_scalar(eq.barrier_dir_deriv(), 0.5 / 4.0 - 1.0 / 8.0 + 1.0 / 2.0 + 2.0 / 10.0));

    resto_ineq_constr iq;
    iq.resize(1, 1);
    iq.t << 11.0, 13.0;
    iq.p << 17.0, 19.0;
    iq.n << 23.0, 29.0;
    iq.t_backup << 31.0, 37.0;
    iq.p_backup << 41.0, 43.0;
    iq.n_backup << 47.0, 53.0;
    iq.d_t << 1.0, -2.0;
    iq.d_p << 3.0, 4.0;
    iq.d_n << -5.0, 6.0;

    REQUIRE(approx_scalar(iq.penalty_sum(), 88.0));
    REQUIRE(approx_scalar(iq.penalty_dir_deriv(), 8.0));
    REQUIRE(approx_scalar(iq.barrier_log_sum(),
                          std::log(11.0) + std::log(13.0) + std::log(17.0) + std::log(19.0) + std::log(23.0) + std::log(29.0)));
    REQUIRE(approx_scalar(iq.barrier_dir_deriv(),
                          1.0 / 31.0 - 2.0 / 37.0 + 3.0 / 41.0 + 4.0 / 43.0 - 5.0 / 47.0 + 6.0 / 53.0));
}

TEST_CASE("restoration objective summary is assembled from elastic blocks") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.rho_eq = 4.0;
    aux.rho_ineq = 6.0;
    aux.mu_bar = 0.5;

    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.p << 2.0;
    aux.elastic_eq.n << 3.0;
    aux.elastic_eq.p_backup << 5.0;
    aux.elastic_eq.n_backup << 7.0;
    aux.elastic_eq.d_p << 0.2;
    aux.elastic_eq.d_n << -0.3;
    aux.elastic_eq.r_c << 1.25;
    aux.elastic_eq.r_p << 0.4;
    aux.elastic_eq.r_n << -0.6;
    aux.elastic_eq.r_s_p << 0.8;
    aux.elastic_eq.r_s_n << -0.2;

    aux.elastic_ineq.resize(1, 0);
    aux.elastic_ineq.t << 11.0;
    aux.elastic_ineq.p << 13.0;
    aux.elastic_ineq.n << 17.0;
    aux.elastic_ineq.t_backup << 19.0;
    aux.elastic_ineq.p_backup << 23.0;
    aux.elastic_ineq.n_backup << 29.0;
    aux.elastic_ineq.d_t << 0.5;
    aux.elastic_ineq.d_p << -0.7;
    aux.elastic_ineq.d_n << 0.9;
    aux.elastic_ineq.r_d << 2.5;
    aux.elastic_ineq.r_t << -1.1;
    aux.elastic_ineq.r_p << 0.3;
    aux.elastic_ineq.r_n << -0.2;
    aux.elastic_ineq.r_s_t << 0.6;
    aux.elastic_ineq.r_s_p << -0.4;
    aux.elastic_ineq.r_s_n << 0.9;

    const auto summary = current_objective_summary(aux);
    REQUIRE(approx_scalar(summary.exact_penalty, 4.0 * (2.0 + 3.0) + 6.0 * (13.0 + 17.0)));
    REQUIRE(approx_scalar(summary.barrier_value,
                          0.5 * (std::log(2.0) + std::log(3.0) + std::log(11.0) + std::log(13.0) + std::log(17.0))));
    REQUIRE(approx_scalar(summary.penalty_dir_deriv, 4.0 * (0.2 - 0.3) + 6.0 * (-0.7 + 0.9)));
    REQUIRE(approx_scalar(summary.barrier_dir_deriv,
                          0.5 * ((0.2 / 5.0) + (-0.3 / 7.0) + (0.5 / 19.0) + (-0.7 / 23.0) + (0.9 / 29.0))));
    REQUIRE(approx_scalar(summary.prim_res_l1, 1.25 + 2.5));
    REQUIRE(approx_scalar(summary.inf_local_stat, 0.6));
    REQUIRE(approx_scalar(summary.inf_local_comp, 0.8));
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
    REQUIRE(residual.w_stationarity[__x].isApprox((row_vector(2) << 1.0, -2.0).finished()));
    REQUIRE(residual.w_stationarity[__u].isApprox((row_vector(2) << 0.5, -0.2).finished()));
    REQUIRE(residual.w_stationarity[__y].isApprox((row_vector(2) << -0.7, 0.3).finished()));
    REQUIRE(approx_scalar(residual.eq_local.inf_prim, 0.6));
    REQUIRE(approx_scalar(residual.eq_local.inf_stat, 0.25));
    REQUIRE(approx_scalar(residual.eq_local.inf_comp, 0.5));
    REQUIRE(approx_scalar(residual.ineq_local.inf_prim, 0.8));
    REQUIRE(approx_scalar(residual.ineq_local.inf_stat, 0.9));
    REQUIRE(approx_scalar(residual.ineq_local.inf_comp, 0.7));
    REQUIRE(approx_scalar(residual.inf_primal, 1.2));
    REQUIRE(approx_scalar(residual.inf_dual, 2.0));
    REQUIRE(approx_scalar(residual.inf_comp, 0.5));
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

TEST_CASE("restoration mu update follows IPM-style complementarity ratio in adaptive mode") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.mu_bar = 10.0;
    aux.elastic_eq.resize(1, 0);
    aux.elastic_eq.p_backup << 2.0;
    aux.elastic_eq.nu_p_backup << 1.0;
    aux.elastic_eq.n_backup << 3.0;
    aux.elastic_eq.nu_n_backup << 2.0;
    aux.elastic_eq.p << 1.0;
    aux.elastic_eq.nu_p << 0.5;
    aux.elastic_eq.n << 2.0;
    aux.elastic_eq.nu_n << 1.0;

    solver::ipm_config cfg;
    cfg.mu_method = solver::ipm_config::mehrotra_predictor_corrector;

    REQUIRE(update_mu_bar(aux, cfg, 10.0, 0.2, 1.0, 1.0));
    const scalar_t prev_aff = 2.0 * 1.0 + 3.0 * 2.0;
    const scalar_t post_aff = 1.0 * 0.5 + 2.0 * 1.0;
    const scalar_t eta = post_aff / prev_aff;
    const scalar_t sig = std::pow(eta, 3);
    REQUIRE(approx_scalar(aux.mu_bar, sig * prev_aff / 2.0));
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

TEST_CASE("restoration predictor bookkeeping follows normal IPM complementarity accounting") {
    solver::linesearch_config cfg;
    cfg.alpha_primal = 0.4;
    cfg.alpha_dual = 0.7;

    resto_elastic_constr eq;
    eq.resize(1, 0);
    eq.p << 2.0;
    eq.n << 3.0;
    eq.nu_p << 5.0;
    eq.nu_n << 7.0;
    eq.d_p << -0.5;
    eq.d_n << 0.25;
    eq.d_nu_p << -1.5;
    eq.d_nu_n << 0.5;

    resto_ineq_constr iq;
    iq.resize(1, 0);
    iq.t << 11.0;
    iq.p << 13.0;
    iq.n << 17.0;
    iq.nu_t << 19.0;
    iq.nu_p << 23.0;
    iq.nu_n << 29.0;
    iq.d_t << -2.0;
    iq.d_p << 1.5;
    iq.d_n << -1.0;
    iq.d_nu_t << -3.0;
    iq.d_nu_p << 2.0;
    iq.d_nu_n << -4.0;

    solver::ipm_config::worker worker;
    eq.finalize_predictor_step(cfg, worker);
    iq.finalize_predictor_step(cfg, worker);

    REQUIRE(worker.n_ipm_cstr == 5);
    const scalar_t alpha_d = cfg.dual_alpha_for_ineq();
    const scalar_t prev_aff =
        2.0 * 5.0 + 3.0 * 7.0 + 11.0 * 19.0 + 13.0 * 23.0 + 17.0 * 29.0;
    const scalar_t post_aff =
        (5.0 + alpha_d * -1.5) * (2.0 + cfg.alpha_primal * -0.5) +
        (7.0 + alpha_d * 0.5) * (3.0 + cfg.alpha_primal * 0.25) +
        (19.0 + alpha_d * -3.0) * (11.0 + cfg.alpha_primal * -2.0) +
        (23.0 + alpha_d * 2.0) * (13.0 + cfg.alpha_primal * 1.5) +
        (29.0 + alpha_d * -4.0) * (17.0 + cfg.alpha_primal * -1.0);
    REQUIRE(approx_scalar(worker.prev_aff_comp, prev_aff));
    REQUIRE(approx_scalar(worker.post_aff_comp, post_aff));
}

TEST_CASE("restoration mu update prefers predictor worker data when available") {
    ns_riccati::ns_riccati_data::restoration_aux_data aux;
    aux.mu_bar = 5.0;
    aux.predictor_worker.n_ipm_cstr = 2;
    aux.predictor_worker.prev_aff_comp = 20.0;
    aux.predictor_worker.post_aff_comp = 5.0;

    solver::ipm_config cfg;
    cfg.mu_method = solver::ipm_config::mehrotra_predictor_corrector;

    REQUIRE(update_mu_bar(aux, cfg, 10.0, 0.2, 1.0, 1.0));
    const scalar_t eta = 5.0 / 20.0;
    const scalar_t sig = std::pow(eta, 3);
    REQUIRE(approx_scalar(aux.mu_bar, sig * 20.0 / 2.0));
    REQUIRE(aux.predictor_worker.n_ipm_cstr == 0);
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
