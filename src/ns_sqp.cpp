#include <atri/solver/ns_riccati_solver.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <atri/solver/ns_riccati_data.hpp>


namespace atri {
ns_sqp::node::node(problem_ptr_t prob)
    : shooting_node(prob, data_mgr::get<nullspace_riccati_data>()) {}
void ns_sqp::update() {
    graph_.apply_all_unary_parallel(ns_riccati_solver::pre_solving_steps_1);
    graph_.apply_all_binary_forward<true>(ns_riccati_solver::pre_solving_steps_2);
    graph_.apply_all_binary_backward<true>(ns_riccati_solver::backward_pass);
    graph_.apply_all_unary_parallel(ns_riccati_solver::post_solving_steps);
    graph_.apply_all_binary_forward<false, true>(ns_riccati_solver::forward_rollout);
    graph_.apply_all_unary_parallel(ns_riccati_solver::post_rollout_steps);
}
} // namespace atri
