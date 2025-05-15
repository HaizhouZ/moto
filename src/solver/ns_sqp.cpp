#include <atri/solver/ns_riccati_data.hpp>
#include <atri/solver/ns_riccati_solve.hpp>
#include <atri/solver/ns_sqp.hpp>

#include <atri/utils/timed_block.hpp>



namespace atri {
void ns_sqp::update(size_t n_iter) {

    // timed_block_labeled("all", {
        timed_block(graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_0));
        for (size_t i_iter : range(n_iter)) {

            timed_block(graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_1));
            timed_block(graph_.apply_all_binary_forward<true>(ns_riccati::pre_solving_steps_2));
            timed_block(graph_.apply_all_binary_backward<true>(ns_riccati::backward_pass));
            timed_block(graph_.apply_all_unary_parallel(ns_riccati::post_solving_steps));
            timed_block(graph_.apply_all_binary_forward<false, true>(ns_riccati::forward_rollout));
            timed_block(graph_.apply_all_unary_parallel(ns_riccati::post_rollout_steps));
            timed_block(graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_0));
        }
    // });
}
} // namespace atri
