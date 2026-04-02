#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>

#define private public
#include <moto/solver/ns_sqp.hpp>
#undef private
#include <moto/ocp/cost.hpp>
#include <moto/solver/restoration/resto_init.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>
#include <moto/solver/restoration/resto_runtime.hpp>

namespace moto {

TEST_CASE("restoration rejects merit backtracking") {
    ns_sqp sqp;
    sqp.settings.ls.method = ns_sqp::linesearch_setting::search_method::merit_backtracking;

    ns_sqp::kkt_info kkt_before;
    ns_sqp::filter_linesearch_data ls;

    REQUIRE_THROWS_WITH(
        sqp.restoration_update(kkt_before, ls),
        Catch::Matchers::ContainsSubstring("restoration mode is incompatible with merit_backtracking"));
}

TEST_CASE("restoration start augments outer filter with relaxed point") {
    ns_sqp sqp;
    sqp.settings.ls.primal_gamma = 0.2;
    sqp.settings.ls.dual_gamma = 0.3;

    ns_sqp::filter_linesearch_data ls;
    ns_sqp::kkt_info ref;
    ref.prim_res_l1 = 2.5;
    ref.inf_dual_res = 7.0;
    ref.objective = 11.0;

    ls.augment_filter_for_restoration_start(ref, sqp.settings);

    REQUIRE(ls.points.size() == 1);
    REQUIRE(std::abs(ls.points.front().prim_res - (1.0 - sqp.settings.ls.primal_gamma) * ref.prim_res_l1) < 1e-12);
    REQUIRE(std::abs(ls.points.front().dual_res - ref.inf_dual_res) < 1e-12);
    REQUIRE(std::abs(ls.points.front().objective - (ref.objective - sqp.settings.ls.dual_gamma * ref.prim_res_l1)) < 1e-12);
}

TEST_CASE("restoration acceptor does not accept merely because outer filter would") {
    ns_sqp sqp;
    sqp.settings.ls.primal_gamma = 0.1;
    sqp.settings.ls.dual_gamma = 1e-4;
    sqp.settings.ls.max_steps = 0;

    ns_sqp::filter_linesearch_data ls;
    ls.resto.current.inf_prim_res = 1.0;
    ls.resto.current.objective = 10.0;
    ls.resto.trial.inf_prim_res = 0.6;
    ls.resto.trial.objective = 10.1;
    ls.resto.trial.inf_dual_res = 0.2;
    ls.resto.trial.inf_comp_res = 0.3;
    ls.resto.points.push_back({.theta = 0.5, .phi = 10.0});

    ns_sqp::kkt_info current;
    current.prim_res_l1 = 1.0;
    current.objective = 10.0;
    current.cost = 10.0;
    current.obj_fullstep_dec = 0.0;
    current.barrier_dir_deriv = 0.0;

    ns_sqp::kkt_info trial = current;
    trial.prim_res_l1 = 0.6;
    trial.objective = 10.1;

    REQUIRE(sqp.outer_filter_accepts(ls, trial, current));
    REQUIRE(sqp.restoration_linesearch(ls, trial, current) == ns_sqp::line_search_action::failure);
}

TEST_CASE("elastic restoration initialization satisfies p-n=c and zp*p=zn*n=mu") {
    constexpr scalar_t c = 0.3;
    constexpr scalar_t rho = 10.0;
    constexpr scalar_t mu_bar = 0.7;
    auto init = solver::restoration::initialize_elastic_pair(c, rho, mu_bar);

    REQUIRE(init.p > 0.);
    REQUIRE(init.n > 0.);
    REQUIRE(std::abs((init.p - init.n) - c) < 1e-12);
    REQUIRE(std::abs(init.z_p * init.p - mu_bar) < 1e-12);
    REQUIRE(std::abs(init.z_n * init.n - mu_bar) < 1e-12);
    REQUIRE(std::abs(init.lambda - (rho - init.z_p)) < 1e-12);
    REQUIRE(std::abs(init.lambda - (init.z_n - rho)) < 1e-12);
}

TEST_CASE("elastic restoration weight matches lambda derivative") {
    constexpr scalar_t c = -0.2;
    constexpr scalar_t rho = 5.0;
    constexpr scalar_t mu_bar = 0.4;
    constexpr scalar_t eps = 1e-7;

    const auto center = solver::restoration::initialize_elastic_pair(c, rho, mu_bar);
    const auto plus = solver::restoration::initialize_elastic_pair(c + eps, rho, mu_bar);
    const auto minus = solver::restoration::initialize_elastic_pair(c - eps, rho, mu_bar);
    const scalar_t fd = (plus.lambda - minus.lambda) / (2. * eps);

    REQUIRE(center.weight > 0.);
    REQUIRE(std::abs(center.weight - fd) < 1e-6);
}

TEST_CASE("explicit elastic local KKT recovery matches direct linear solve") {
    constexpr scalar_t c = 0.3;
    constexpr scalar_t rho = 5.0;
    constexpr scalar_t mu_bar = 0.7;
    constexpr scalar_t lambda_reg = 1e-3;
    constexpr scalar_t delta_c = 0.17;

    const auto init = solver::restoration::initialize_elastic_pair(c, rho, mu_bar);

    resto_elastic_constr elastic;
    elastic.resize(1, 0);
    elastic.p(0) = init.p * 1.1;
    elastic.n(0) = init.n * 0.9;
    elastic.nu_p(0) = init.z_p * 0.95;
    elastic.nu_n(0) = init.z_n * 1.05;

    vector c_vec(1);
    c_vec << c;
    vector lambda(1);
    lambda << init.lambda + 0.11;

    solver::restoration::compute_local_model(c_vec, lambda, elastic, rho, mu_bar, lambda_reg);

    vector delta_c_vec(1);
    delta_c_vec << delta_c;
    solver::restoration::recover_local_step(delta_c_vec, elastic, lambda_reg);

    matrix A(5, 5);
    A << -1., 1., 0., -lambda_reg, 0.,
         1., 0., 1., 0., -lambda_reg,
         0., elastic.nu_p(0), 0., elastic.p(0), 0.,
         0., 0., elastic.nu_n(0), 0., elastic.n(0),
         0., -1., 1., 0., 0.;

    vector rhs(5);
    rhs << -elastic.r_p(0),
           -elastic.r_n(0),
           -elastic.r_s_p(0),
           -elastic.r_s_n(0),
           -(elastic.r_c(0) + delta_c);

    const vector direct = A.fullPivLu().solve(rhs);

    REQUIRE(std::abs(elastic.d_lambda(0) - direct(0)) < 1e-10);
    REQUIRE(std::abs(elastic.d_p(0) - direct(1)) < 1e-10);
    REQUIRE(std::abs(elastic.d_n(0) - direct(2)) < 1e-10);
    REQUIRE(std::abs(elastic.d_nu_p(0) - direct(3)) < 1e-10);
    REQUIRE(std::abs(elastic.d_nu_n(0) - direct(4)) < 1e-10);
}

TEST_CASE("explicit elastic local KKT recovery matches direct linear solve without regularization") {
    constexpr scalar_t c = -0.25;
    constexpr scalar_t rho = 4.0;
    constexpr scalar_t mu_bar = 0.6;
    constexpr scalar_t lambda_reg = 0.0;
    constexpr scalar_t delta_c = -0.09;

    const auto init = solver::restoration::initialize_elastic_pair(c, rho, mu_bar);

    resto_elastic_constr elastic;
    elastic.resize(1, 0);
    elastic.p(0) = init.p * 0.8;
    elastic.n(0) = init.n * 1.2;
    elastic.nu_p(0) = init.z_p * 1.1;
    elastic.nu_n(0) = init.z_n * 0.9;

    vector c_vec(1);
    c_vec << c;
    vector lambda(1);
    lambda << init.lambda - 0.07;

    solver::restoration::compute_local_model(c_vec, lambda, elastic, rho, mu_bar, lambda_reg);

    vector delta_c_vec(1);
    delta_c_vec << delta_c;
    solver::restoration::recover_local_step(delta_c_vec, elastic, lambda_reg);

    matrix A(5, 5);
    A << -1., 0., 0., -1., 0.,
         1., 0., 0., 0., -1.,
         0., elastic.nu_p(0), 0., elastic.p(0), 0.,
         0., 0., elastic.nu_n(0), 0., elastic.n(0),
         0., -1., 1., 0., 0.;

    vector rhs(5);
    rhs << -elastic.r_p(0),
           -elastic.r_n(0),
           -elastic.r_s_p(0),
           -elastic.r_s_n(0),
           -(elastic.r_c(0) + delta_c);

    const vector direct = A.fullPivLu().solve(rhs);

    REQUIRE(std::abs(elastic.d_lambda(0) - direct(0)) < 1e-10);
    REQUIRE(std::abs(elastic.d_p(0) - direct(1)) < 1e-10);
    REQUIRE(std::abs(elastic.d_n(0) - direct(2)) < 1e-10);
    REQUIRE(std::abs(elastic.d_nu_p(0) - direct(3)) < 1e-10);
    REQUIRE(std::abs(elastic.d_nu_n(0) - direct(4)) < 1e-10);
}

TEST_CASE("explicit elastic restoration bounds cap primal and dual steps") {
    resto_elastic_constr elastic;
    elastic.resize(2, 0);
    elastic.p << 1.0, 0.5;
    elastic.n << 0.8, 0.4;
    elastic.nu_p << 2.0, 3.0;
    elastic.nu_n << 4.0, 5.0;
    elastic.d_p << -2.0, 0.1;
    elastic.d_n << -1.0, -0.2;
    elastic.d_nu_p << -3.0, 0.2;
    elastic.d_nu_n << 0.1, -10.0;

    solver::linesearch_config ls;
    elastic.update_ls_bounds(ls, 0.9);

    REQUIRE(ls.primal.alpha_max <= 0.72 + 1e-12);
    REQUIRE(ls.dual.alpha_max <= 0.45 + 1e-12);
}

TEST_CASE("restoration prox term does not modify outer cost bookkeeping") {
    auto [x, y] = sym::states("x_resto_prox_unit", 1);
    auto u = sym::inputs("u_resto_prox_unit", 1);
    auto prob = ocp::create();
    prob->add(*cost(new generic_cost(
        "cost_resto_prox_unit",
        var_list{x, u, y},
        x + u + y,
        approx_order::second)));

    ns_sqp::data d(prob);
    d.aux_.reset(new solver::ns_riccati::ns_riccati_data::restoration_aux_data());

    d.sym_val().value_[__u](0) = 2.0;
    d.sym_val().value_[__y](0) = -3.0;
    d.restoration_prox_.u_ref = vector::Constant(1, 0.5);
    d.restoration_prox_.y_ref = vector::Constant(1, -1.0);
    d.restoration_prox_.sigma_u_sq = vector::Constant(1, 4.0);
    d.restoration_prox_.sigma_y_sq = vector::Constant(1, 9.0);

    d.dense().cost_ = 7.0;
    d.dense().cost_jac_[__u] = row_vector::Constant(1, 11.0);
    d.dense().cost_jac_[__y] = row_vector::Constant(1, 13.0);
    d.dense().lag_ = 0.0;
    d.dense().lag_jac_[__u].setZero();
    d.dense().lag_jac_[__y].setZero();

    solver::restoration::add_resto_prox_term(d, 2.0, 3.0);

    REQUIRE(d.dense().cost_ == Catch::Approx(7.0));
    REQUIRE(d.dense().cost_jac_[__u](0) == Catch::Approx(11.0));
    REQUIRE(d.dense().cost_jac_[__y](0) == Catch::Approx(13.0));

    const scalar_t expected_lag =
        0.5 * 2.0 * 4.0 * std::pow(2.0 - 0.5, 2) +
        0.5 * 3.0 * 9.0 * std::pow(-3.0 - (-1.0), 2);
    REQUIRE(d.dense().lag_ == Catch::Approx(expected_lag));
    REQUIRE(d.dense().lag_jac_[__u](0) == Catch::Approx(2.0 * 4.0 * (2.0 - 0.5)));
    REQUIRE(d.dense().lag_jac_[__y](0) == Catch::Approx(3.0 * 9.0 * (-3.0 - (-1.0))));
}

} // namespace moto
