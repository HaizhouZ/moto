#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#define private public
#include <moto/solver/ns_sqp.hpp>
#undef private
#include <moto/solver/restoration/resto_bound_constr.hpp>
#include <moto/solver/restoration/resto_init.hpp>

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

TEST_CASE("resto bound constr creates solver-managed __s storage") {
    resto_bound_constr b(resto_bound_type::positive, 4, "resto_p");
    REQUIRE(b.field() == __ineq_x);
    REQUIRE(b.dim() == 4);
    REQUIRE(b.in_args().size() == 1);
    REQUIRE(b.in_args().front()->field() == __s);

    auto *cloned_expr = b.clone();
    auto *cloned = dynamic_cast<resto_bound_constr *>(cloned_expr);
    REQUIRE(cloned != nullptr);
    REQUIRE(cloned->field() == __ineq_x);
    REQUIRE(cloned->dim() == 4);
    REQUIRE(cloned->in_args().size() == 1);
    REQUIRE(cloned->in_args().front()->field() == __s);
    delete cloned_expr;
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

} // namespace moto
