#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/core/expr.hpp>
#include <moto/ocp/impl/func.hpp>

TEST_CASE("exprOwnership") {
    using namespace moto;
    sym a("a", 2, __x);
    sym b("b", 3, __x);
    a.add_dep(b); // add b as a dependency of a
    func c("c", {b}, cs::SX::sym("c_out"), approx_order::first);
    c.add_argument(a);
    assert(c.in_args().size() == 2 && "Function should have 2 input argument");
    assert(a.shared() && "Symbol a should be shared");
    assert(b.shared() && "Symbol a should be shared");
    assert(!c.shared() && "Function c should be not shared");
}