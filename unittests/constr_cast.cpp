#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/ineq_constr.hpp>

struct test_ineq : public moto::generic_constr {
    test_ineq(moto::generic_constr &&rhs) : moto::generic_constr(std::move(rhs)) {
        // Custom initialization for test_ineq if needed
    }
    test_ineq(const moto::generic_constr &rhs) : moto::generic_constr(rhs) {
        // Custom initialization for test_ineq if needed
    }
};

TEST_CASE("exprOwnership") {
    using namespace moto;
    auto [a, b] = sym::states("a", 3);
    auto c = constr("a_plus_b", {a, b}, a + b, approx_order::first);
    c->finalize();
    auto p = c.cast_ineq<test_ineq>();
    assert(p->finalized());
    auto ip = p->clone().cast_ineq<test_ineq>();
    assert(!ip->finalized());
}