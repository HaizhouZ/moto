#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <atri/eval/external_function.hpp>
#include <iostream>

TEST_CASE("externalFunc") {
    using namespace atri;
    auto func = load_from_shared<vector, vector>("gen/librnea.so", "rnea");
    auto q = vector(12).setRandom();
    auto v = vector(12).setRandom();
    auto a = vector(12).setRandom();
    auto tq = vector(12).setZero();
    std::vector<vector_ref> input_refs{q, v, a};
    std::vector<vector_ref> output_refs{tq};
    func(input_refs, output_refs);
    std::cout << tq.transpose() << '\n';
}