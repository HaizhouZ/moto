#include <atri/solver/ns_riccati_data.hpp>
#include <atri/solver/ns_riccati_solver.hpp>
#include <atri/solver/ns_sqp.hpp>

namespace atri {
ns_sqp::node::node(problem_ptr_t prob)
    : shooting_node(prob, data_mgr::get<nullspace_riccati_data>()) {}
void ns_sqp::update(size_t n_iter) {

    auto start = std::chrono::high_resolution_clock::now();

    graph_.apply_all_unary_parallel(ns_riccati_solver::pre_solving_steps_0);
    timings[0] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

    for (size_t i_iter: range(n_iter)) {

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_unary_parallel(ns_riccati_solver::pre_solving_steps_1);
        timings[1] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_binary_forward<true>(ns_riccati_solver::pre_solving_steps_2);
        timings[2] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_binary_backward<true>(ns_riccati_solver::backward_pass);
        timings[3] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_unary_parallel(ns_riccati_solver::post_solving_steps);
        timings[4] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_binary_forward<false, true>(ns_riccati_solver::forward_rollout);
        timings[5] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        graph_.apply_all_unary_parallel(ns_riccati_solver::post_rollout_steps);
        timings[6] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        graph_.apply_all_unary_parallel(ns_riccati_solver::pre_solving_steps_0);
        timings[0] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    }
}
} // namespace atri
