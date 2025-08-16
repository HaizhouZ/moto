#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/field_conversion.hpp>
#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>
#include <thread>
TEST_CASE("field_conversion") {
    using namespace moto;
    auto d1 = constr("d1", approx_order::first, 2, __dyn);
    auto d2 = constr("d2", approx_order::first, 2, __dyn);
    auto [x, y] = sym::states("x", 2);
    auto [x2, y2] = sym::states("x2", 2);
    d1->add_arguments({x, y});
    d2->add_arguments({x2, y2});

    auto prob_x = ocp::create();
    prob_x->add(d1);
    prob_x->add(d2);

    auto prob_y = ocp::create();
    prob_y->add(d2);
    prob_y->add(d1);

    { // copy x of prob_x to y of prob_y
        auto x_data = vector::Random(prob_x->dim(__x)).eval();
        auto y_data = vector::Zero(prob_y->dim(__y)).eval();

        utils::copy_x_to_y(x_data, y_data, prob_x.get(), prob_y.get());
        REQUIRE(y_data.head(d2->dim()) == x_data.tail(d2->dim()));
        REQUIRE(y_data.tail(d1->dim()) == x_data.head(d1->dim()));
    }
    {
        // copy y of prob_y to x of prob_x
        auto y_data = vector::Random(prob_y->dim(__y)).eval();
        auto x_data = vector::Zero(prob_x->dim(__x)).eval();

        utils::copy_y_to_x(y_data, x_data, prob_y.get(), prob_x.get());
        REQUIRE(y_data.head(d2->dim()) == x_data.tail(d2->dim()));
        REQUIRE(y_data.tail(d1->dim()) == x_data.head(d1->dim()));
    }
    
    { // copy x of prob_x to y of prob_y with permutation
        auto x_data = vector::Random(prob_x->dim(__x)).eval();
        auto y_data = vector::Zero(prob_y->dim(__y)).eval();
        y_data.head(d2->dim()) = x_data.tail(d2->dim());
        y_data.tail(d1->dim()) = x_data.head(d1->dim());

        auto &perm = utils::permutation_from_y_to_x(prob_y.get(), prob_x.get());
        REQUIRE(perm * y_data == x_data);
    }
}