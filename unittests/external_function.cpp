#define CATCH_CONFIG_MAIN
#include <atri/core/external_function.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <filesystem>

TEST_CASE("externalFunc") {
    using namespace atri;
    std::cout << "Current directory: " << std::filesystem::current_path() << '\n';
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