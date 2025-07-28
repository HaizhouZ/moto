#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/core/expr.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/utils/codegen.hpp>

using namespace moto;

TEST_CASE("symbolicComp") {
    sym a("a", 3, __x);
    sym b("b", 3, __x);
    // sym c;
    auto c = cs::SX::dot(a, b);
    std::cout << c << '\n';
    generate_and_compile("test",
                         codegen_opts{
                             .sx_inputs = {a, b},
                             .compile = true,
                             //  .gen_eval=false,
                             .gen_jacobian = true,
                             .gen_hessian = true,
                             .check_jac_ad = true,
                         },
                         c);
    wait_until_generated();
    // auto c = a + b;
    // c = a;
    // c(0) = cs::SX::dot(a, b);
    // // auto f = cs::Function("test", {a, b}, {c});
    // // f.save("gen/test");
    // auto cstr = generic_constr("cdotab", {a, b}, c, __eq_xu, approx_order::first);
    // std::cout << c.serialize() << '\n';
    // std::cout << a << '\n';
    // std::cout << c << '\n';
}