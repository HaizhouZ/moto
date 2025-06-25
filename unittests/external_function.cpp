#define CATCH_CONFIG_MAIN
#include <moto/core/external_function.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>

TEST_CASE("externalFunc") {
    using namespace moto;
    std::cout << "Current directory: " << std::filesystem::current_path() << '\n';
    auto func = ext_func("gen/librnea_hess.so", "rnea_hess");
    auto q = vector(7).setRandom();
    auto v = vector(7).setRandom();
    auto vn = vector(7).setRandom();
    auto tq = vector(7).setZero();
    auto lbd = vector(7).setZero();

    std::vector<vector_ref> input_refs{q, v, vn, tq, lbd};
    std::vector<std::vector<matrix>> hess;
    std::vector<std::vector<matrix_ref>> output_refs;
    for (auto &ref : input_refs) {
        hess.push_back(std::vector<matrix>(7, matrix(7, 7).setZero()));
        output_refs.push_back(std::vector<matrix_ref>());
        size_t i = 0;
        for (auto &ref2 : input_refs) {
            output_refs.back().push_back(hess.back()[i]);
            i++;
        }
    }
    // Benchmark the function invocation
    const int num_iters = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        func.invoke(input_refs, output_refs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "Average invoke time: " << (elapsed.count() / num_iters) << " us\n";
    std::cout << tq.transpose() << '\n';
}