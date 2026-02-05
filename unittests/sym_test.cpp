#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/core/expr.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

int main() {
    using namespace moto;
    auto default_value = (vector(3) << 0.1, 0.2, 0.3).finished();
    auto u = moto::sym::inputs("u", 3, default_value);
    std::cout << u << std::endl;
    cs::SX::sin(u);
    // REQUIRE(u->name() == "u");
    // REQUIRE(u->dim() == 3);
    // REQUIRE(u->field() == __u);
    // REQUIRE(u->default_value().isApprox(default_value));
    return 0;
}