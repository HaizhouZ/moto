#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#define private public
#include <moto/solver/ns_sqp.hpp>
#undef private
#include <moto/solver/restoration/resto_init.hpp>
#include <moto/solver/restoration/resto_elastic_constr.hpp>

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

} // namespace moto
