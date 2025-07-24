#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/constr.hpp>

TEST_CASE("exprCast") {
    using namespace moto;
    sym a("a", 3, __x);
    sym b("b", 3, __x);
    a.add_dep(b);      // add b as a dependency of a
    cs::SX c_ = a + b; // convert sym to SX
    func c("c", {b}, c_, approx_order::first, __eq_x);
    c.cast<constr>();
}