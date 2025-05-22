#define CATCH_CONFIG_MAIN
#include <atri/core/expr.hpp>
#include <atri/ocp/constr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>

using namespace atri;

TEST_CASE("symbolicComp") {
    sym a("a", 3, __x);
    sym b("b", 3, __x);
    sym c;
    c = a;
    c(0) = cs::SX::dot(a, b);
    // auto f = cs::Function("test", {a, b}, {c});
    // f.save("gen/test");
    auto cstr = constr("cdotab", {a, b}, c, __eq_cstr_c, approx_order::first);
    std::cout << c.serialize() << '\n';
    std::cout << a << '\n';
    std::cout << c << '\n';
}