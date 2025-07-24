#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/core/expr.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>

TEST_CASE("exprOwnership") {
    using namespace moto;
    sym a("a", 3, __x);
    sym b("b", 3, __x);
    a.add_dep(b);      // add b as a dependency of a
    cs::SX c_ = a + b; // convert sym to SX
    std::cout << "a is: " << a << std::endl;
    std::cout << "b is: " << b << std::endl;
    std::cout << "c_ is: " << c_ << std::endl;
    func c("c", {b}, c_, approx_order::first, __eq_x);
    c.add_argument(a);
    assert(c.in_args().size() == 2 && "Function should have 2 input argument");
    std::cout << "a has uid: " << a.uid() << " and b has uid: " << b.uid() << std::endl;
    for (sym &arg : c.dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    auto d = c;
    for (sym &arg : d.dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    sym e("e", 3, __x);
    d.add_dep(e); // add a as a dependency of d
    for (sym &arg : c.dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    sym f("f", 3, __x);
    auto p = d.clone<func>(); // clone d to p
    p.add_argument(f);        // add f as an argument to p
    for (sym &arg : c.dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    for (sym &arg : p.dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    auto prob = ocp::create();
    prob->add(c);
    for (const func &f : prob->exprs(__eq_x)) {
        std::cout << "Function in problem: " << f.name() << " with uid: " << f.uid() << std::endl;
    }
    for (const sym &arg : prob->exprs(__x)) {
        std::cout << "Symbol in problem: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
}