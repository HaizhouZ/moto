#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/core/expr.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

TEST_CASE("exprOwnership") {
    using namespace moto;
    var a = sym::state("a", 3);
    var b = a->next();
    a->next() = b;
    a->add_dep(b);   // add b as a dependency of a
    auto c_ = a + b; // convert sym to SX
    std::cout << "a is: " << a << std::endl;
    std::cout << "b is: " << b << std::endl;
    std::cout << "c_ is: " << c_ << std::endl;
    func c = constr("c", {a, b}, c_, approx_order::first, __eq_x);
    assert(c->in_args().size() == 2 && "Function should have 2 input argument");
    std::cout << "a has uid: " << a->uid() << " and b has uid: " << b->uid() << std::endl;
    for (sym &arg : c->dep()) {
        std::cout << "Function argument: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
    auto d = c; // shallow copy
    assert(d->in_args().size() == c->in_args().size() && "Function d should have same input arguments as c");
    {
        size_t d_arg_idx = 0;
        for (sym &arg : d->dep()) {
            assert(arg.uid() == c->in_args()[d_arg_idx++]->uid() &&
                   "Function d should have same input arguments as c");
        }
    }
    var e = sym::inputs("e", 3);
    {
        d->add_argument(e); // add a as a dependency of d
        size_t d_arg_idx = 0;
        assert(d->in_args().size() == c->in_args().size() && "Function d should have same input arguments as c");
        for (sym &arg : d->dep()) {
            assert(arg.uid() == c->in_args()[d_arg_idx++]->uid() &&
                   "Function d should have same input arguments as c");
        }
    }
    var f = sym::params("f", 3);
    auto p = d->clone(); // clone d to p
    p->add_argument(f);  // add f as an argument to p
    assert(p->in_args().size() == c->in_args().size() + 1 && "Function p should have one more input argument than c");
    auto prob = ocp::create();
    prob->add(c);
    for (const generic_func &f : prob->exprs(__eq_x)) {
        std::cout << "Function in problem: " << f.name() << " with uid: " << f.uid() << std::endl;
    }
    for (const sym &arg : prob->exprs(__x)) {
        std::cout << "Symbol in problem: " << arg.name() << " with uid: " << arg.uid() << std::endl;
    }
}