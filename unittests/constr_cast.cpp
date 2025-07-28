#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/ineq_constr.hpp>

struct test_ineq : public moto::constr {
    test_ineq(moto::constr &&rhs) : moto::constr(std::move(rhs)) {
        // Custom initialization for test_ineq if needed
    }
    test_ineq(const moto::constr &rhs) : moto::constr(rhs) {
        // Custom initialization for test_ineq if needed
    }
};

TEST_CASE("exprOwnership") {
    using namespace moto;
    var a = sym("a", 3, __x);
    var b = sym("b", 3, __y);
    func c = constr("a_plus_b", {a, b}, a + b, approx_order::first).as_eq();
    c->finalize();
    func p = c.as<constr>().as_ineq<test_ineq>();
    assert(p->finalized());
    func ip = p->clone().as<constr>().as_ineq<test_ineq>();
    assert(!ip->finalized());
}