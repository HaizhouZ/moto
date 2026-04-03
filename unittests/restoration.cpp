#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>

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

void resize_eq_state(restoration::detail::eq_local_state &state, size_t ns_dim, size_t nc_dim) {
    auto resize_zero = [](vector &v, Eigen::Index n) {
        v.resize(n);
        v.setZero();
    };
    state.ns = ns_dim;
    state.nc = nc_dim;
    const auto dim_eig = static_cast<Eigen::Index>(state.ns + state.nc);
    for (auto *v : {&state.p, &state.p_backup, &state.d_p, &state.nu_p, &state.nu_p_backup, &state.d_nu_p,
                    &state.n, &state.n_backup, &state.d_n, &state.nu_n, &state.nu_n_backup, &state.d_nu_n,
                    &state.c_current, &state.r_c, &state.r_p, &state.r_n, &state.r_s_p, &state.r_s_n,
                    &state.combo_p, &state.combo_n, &state.b_c, &state.minv_diag, &state.minv_bc, &state.d_lambda,
                    &state.corrector_p, &state.corrector_n}) {
        resize_zero(*v, dim_eig);
    }
}

void resize_ineq_state(restoration::detail::ineq_local_state &state, size_t nx_dim, size_t nu_dim) {
    auto resize_zero = [](vector &v, Eigen::Index n) {
        v.resize(n);
        v.setZero();
    };
    state.nx = nx_dim;
    state.nu = nu_dim;
    const auto dim_eig = static_cast<Eigen::Index>(state.nx + state.nu);
    for (auto *v : {&state.t, &state.t_backup, &state.d_t, &state.nu_t, &state.nu_t_backup, &state.d_nu_t,
                    &state.p, &state.p_backup, &state.d_p, &state.nu_p, &state.nu_p_backup, &state.d_nu_p,
                    &state.n, &state.n_backup, &state.d_n, &state.nu_n, &state.nu_n_backup, &state.d_nu_n,
                    &state.g_current, &state.r_d, &state.r_p, &state.r_n, &state.r_s_t, &state.r_s_p, &state.r_s_n,
                    &state.denom_t, &state.denom_p, &state.denom_n,
                    &state.combo_t, &state.combo_p, &state.combo_n, &state.b_d, &state.minv_diag, &state.minv_bd,
                    &state.corrector_t, &state.corrector_p, &state.corrector_n}) {
        resize_zero(*v, dim_eig);
    }
}
} // namespace

TEST_CASE("restoration equality local KKT recovery satisfies regularized linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 2);
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

    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar, lambda_reg);

    vector delta_c(3);
    delta_c << -0.11, 0.07, 0.19;
    resto_eq_elastic_constr::recover_local_step(delta_c, eq, lambda_reg);

    const vector res_c = delta_c - eq.d_p + eq.d_n + eq.r_c;
    const vector res_p = eq.d_p - eq.d_lambda - lambda_reg * eq.d_nu_p + eq.r_p;
    const vector res_n = eq.d_n + eq.d_lambda - lambda_reg * eq.d_nu_n + eq.r_n;
    const vector res_sp = eq.nu_p.cwiseProduct(eq.d_p) + eq.p.cwiseProduct(eq.d_nu_p) + eq.r_s_p;
    const vector res_sn = eq.nu_n.cwiseProduct(eq.d_n) + eq.n.cwiseProduct(eq.d_nu_n) + eq.r_s_n;
    const auto summary = resto_eq_elastic_constr::linearized_newton_residuals(delta_c, eq, lambda_reg);

    REQUIRE(approx_zero(res_c));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
    REQUIRE(summary.inf_prim < 1e-12);
    REQUIRE(summary.inf_stat < 1e-12);
    REQUIRE(summary.inf_comp < 1e-12);
}

TEST_CASE("restoration equality local KKT recovery satisfies affine predictor linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    eq.p << 0.7, 0.9;
    eq.n << 0.8, 1.1;
    eq.nu_p << 1.3, 0.7;
    eq.nu_n << 0.6, 1.2;

    vector c(2);
    c << 0.2, -0.4;
    vector lambda(2);
    lambda << 0.1, -0.2;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;
    const scalar_t lambda_reg = 1e-3;

    vector mu_p = vector::Zero(2);
    vector mu_n = vector::Zero(2);
    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar, &mu_p, &mu_n, lambda_reg);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    resto_eq_elastic_constr::recover_local_step(delta_c, eq, lambda_reg);

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

TEST_CASE("restoration equality local KKT recovery satisfies corrector linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    eq.p << 0.7, 0.9;
    eq.n << 0.8, 1.1;
    eq.nu_p << 1.3, 0.7;
    eq.nu_n << 0.6, 1.2;

    vector c(2);
    c << 0.2, -0.4;
    vector lambda(2);
    lambda << 0.1, -0.2;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;
    const scalar_t lambda_reg = 1e-3;

    vector mu_p(2), mu_n(2);
    mu_p << 0.25, 0.28;
    mu_n << 0.27, 0.29;
    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar, &mu_p, &mu_n, lambda_reg);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    resto_eq_elastic_constr::recover_local_step(delta_c, eq, lambda_reg);

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
    detail::ineq_local_state iq;
    resize_ineq_state(iq, 2, 1);
    iq.t << 1.1, 0.8, 1.4;
    iq.p << 0.8, 1.0, 1.3;
    iq.n << 0.9, 0.7, 1.2;
    iq.nu_t << 0.4, 0.6, 0.5;
    iq.nu_p << 0.9, 0.8, 0.7;
    iq.nu_n << 0.6, 0.9, 0.5;

    vector g(3);
    g << -0.2, 0.5, -0.1;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;
    const scalar_t lambda_reg = 1e-3;

    resto_ineq_elastic_ipm_constr::compute_local_model(iq, g, rho, mu_bar, lambda_reg);

    vector delta_g(3);
    delta_g << 0.08, -0.12, 0.04;
    resto_ineq_elastic_ipm_constr::recover_local_step(delta_g, iq, lambda_reg);

    const vector res_d = delta_g + iq.d_t - iq.d_p + iq.d_n + iq.r_d;
    const vector res_p = iq.d_p - iq.d_nu_t - lambda_reg * iq.d_nu_p + iq.r_p;
    const vector res_n = iq.d_n + iq.d_nu_t - lambda_reg * iq.d_nu_n + iq.r_n;
    const vector res_st = iq.nu_t.cwiseProduct(iq.d_t) + iq.t.cwiseProduct(iq.d_nu_t) + iq.r_s_t;
    const vector res_sp = iq.nu_p.cwiseProduct(iq.d_p) + iq.p.cwiseProduct(iq.d_nu_p) + iq.r_s_p;
    const vector res_sn = iq.nu_n.cwiseProduct(iq.d_n) + iq.n.cwiseProduct(iq.d_nu_n) + iq.r_s_n;
    const auto summary = resto_ineq_elastic_ipm_constr::linearized_newton_residuals(delta_g, iq, lambda_reg);

    REQUIRE(approx_zero(res_d));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_st));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
    REQUIRE(summary.inf_prim < 1e-12);
    REQUIRE(summary.inf_stat < 1e-12);
    REQUIRE(summary.inf_comp < 1e-12);
}

TEST_CASE("restoration inequality initialization follows slack-plus-elastic centering") {
    const scalar_t g = -0.25;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.2;
    const scalar_t nu_t0 = 0.7;

    const auto init = resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(g, rho, mu_bar, nu_t0);

    REQUIRE(approx_scalar(init.nu_t, nu_t0));
    REQUIRE(approx_scalar(init.t, mu_bar / nu_t0));
    REQUIRE(init.p > 0.);
    REQUIRE(init.n > 0.);
    REQUIRE(init.nu_p > 0.);
    REQUIRE(init.nu_n > 0.);
    REQUIRE(approx_scalar(init.nu_p * init.p, mu_bar, 1e-10));
    REQUIRE(approx_scalar(init.nu_n * init.n, mu_bar, 1e-10));
    REQUIRE(approx_scalar(g + init.t - init.p + init.n, 0., 1e-10));
}

TEST_CASE("restoration inequality initializer centers primal and complementarity") {
    const scalar_t g = -0.35;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;
    const scalar_t nu_t0 = 0.8;

    const auto init = resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(g, rho, mu_bar, nu_t0);

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 0);
    iq.t << init.t;
    iq.p << init.p;
    iq.n << init.n;
    iq.nu_t << init.nu_t;
    iq.nu_p << init.nu_p;
    iq.nu_n << init.nu_n;

    vector g_vec(1);
    g_vec << g;
    resto_ineq_elastic_ipm_constr::compute_local_model(iq, g_vec, rho, mu_bar, 0.0);

    REQUIRE(approx_zero(iq.r_d, 1e-10));
    REQUIRE(approx_zero(iq.r_s_t, 1e-10));
    REQUIRE(approx_zero(iq.r_s_p, 1e-10));
    REQUIRE(approx_zero(iq.r_s_n, 1e-10));
    REQUIRE(iq.r_p.allFinite());
    REQUIRE(iq.r_n.allFinite());
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

TEST_CASE("restoration elastic blocks own their penalty and barrier bookkeeping") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    eq.p << 2.0, 3.0;
    eq.n << 5.0, 7.0;
    eq.p_backup << 4.0, 8.0;
    eq.n_backup << 2.0, 10.0;
    eq.d_p << 0.5, -1.0;
    eq.d_n << 1.0, 2.0;

    REQUIRE(approx_scalar(eq.p.sum() + eq.n.sum(), 17.0));
    REQUIRE(approx_scalar(eq.d_p.sum() + eq.d_n.sum(), 2.5));
    REQUIRE(approx_scalar(eq.p.array().log().sum() + eq.n.array().log().sum(),
                          std::log(2.0) + std::log(3.0) + std::log(5.0) + std::log(7.0)));
    REQUIRE(approx_scalar((eq.d_p.array() / eq.p_backup.array()).sum() + (eq.d_n.array() / eq.n_backup.array()).sum(),
                          0.5 / 4.0 - 1.0 / 8.0 + 1.0 / 2.0 + 2.0 / 10.0));

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 1);
    iq.t << 11.0, 13.0;
    iq.p << 17.0, 19.0;
    iq.n << 23.0, 29.0;
    iq.t_backup << 31.0, 37.0;
    iq.p_backup << 41.0, 43.0;
    iq.n_backup << 47.0, 53.0;
    iq.d_t << -2.0, 1.0;
    iq.d_p << 3.0, 4.0;
    iq.d_n << -5.0, 6.0;

    REQUIRE(approx_scalar(iq.p.sum() + iq.n.sum(), 88.0));
    REQUIRE(approx_scalar(iq.d_p.sum() + iq.d_n.sum(), 8.0));
    REQUIRE(approx_scalar(iq.t.array().log().sum() + iq.p.array().log().sum() + iq.n.array().log().sum(),
                          std::log(11.0) + std::log(13.0) + std::log(17.0) + std::log(19.0) + std::log(23.0) + std::log(29.0)));
    REQUIRE(approx_scalar((iq.d_t.array() / iq.t_backup.array()).sum() +
                              (iq.d_p.array() / iq.p_backup.array()).sum() +
                              (iq.d_n.array() / iq.n_backup.array()).sum(),
                          -2.0 / 31.0 + 1.0 / 37.0 + 3.0 / 41.0 + 4.0 / 43.0 - 5.0 / 47.0 + 6.0 / 53.0));
}

TEST_CASE("restoration predictor bookkeeping follows normal IPM complementarity accounting") {
    solver::linesearch_config cfg;
    cfg.alpha_primal = 0.4;
    cfg.alpha_dual = 0.7;

    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 0);
    eq.p << 2.0;
    eq.n << 3.0;
    eq.nu_p << 5.0;
    eq.nu_n << 7.0;
    eq.d_p << -0.5;
    eq.d_n << 0.25;
    eq.d_nu_p << -1.5;
    eq.d_nu_n << 0.5;

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 0);
    iq.t << 11.0;
    iq.p << 13.0;
    iq.n << 17.0;
    iq.nu_t << 19.0;
    iq.nu_p << 23.0;
    iq.nu_n << 29.0;
    iq.d_t << 0.5;
    iq.d_p << 1.5;
    iq.d_n << -1.0;
    iq.d_nu_t << -3.0;
    iq.d_nu_p << 2.0;
    iq.d_nu_n << -4.0;

    solver::ipm_config::worker worker;
    const scalar_t alpha_d = cfg.dual_alpha_for_ineq();
    for (const auto &[value, step, dual, dual_step] : {
             std::tuple{&eq.p, &eq.d_p, &eq.nu_p, &eq.d_nu_p},
             std::tuple{&eq.n, &eq.d_n, &eq.nu_n, &eq.d_nu_n},
             std::tuple{&iq.t, &iq.d_t, &iq.nu_t, &iq.d_nu_t},
             std::tuple{&iq.p, &iq.d_p, &iq.nu_p, &iq.d_nu_p},
             std::tuple{&iq.n, &iq.d_n, &iq.nu_n, &iq.d_nu_n},
         }) {
        worker.n_ipm_cstr += static_cast<size_t>(value->size());
        worker.prev_aff_comp += dual->dot(*value);
        worker.post_aff_comp +=
            (*dual + alpha_d * *dual_step).dot(*value + cfg.alpha_primal * *step);
    }

    REQUIRE(worker.n_ipm_cstr == 5);
    const scalar_t prev_aff =
        2.0 * 5.0 + 3.0 * 7.0 + 11.0 * 19.0 + 13.0 * 23.0 + 17.0 * 29.0;
    const scalar_t post_aff =
        (5.0 + alpha_d * -1.5) * (2.0 + cfg.alpha_primal * -0.5) +
        (7.0 + alpha_d * 0.5) * (3.0 + cfg.alpha_primal * 0.25) +
        (19.0 + alpha_d * -3.0) * (11.0 + cfg.alpha_primal * 0.5) +
        (23.0 + alpha_d * 2.0) * (13.0 + cfg.alpha_primal * 1.5) +
        (29.0 + alpha_d * -4.0) * (17.0 + cfg.alpha_primal * -1.0);
    REQUIRE(approx_scalar(worker.prev_aff_comp, prev_aff));
    REQUIRE(approx_scalar(worker.post_aff_comp, post_aff));
}
