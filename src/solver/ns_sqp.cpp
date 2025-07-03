#include <moto/solver/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/ns_sqp.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
void ns_sqp::forward() {
    timed_block(graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_0));
}
void ns_sqp::update(size_t n_iter) {

    // timed_block_labeled("all", {
    // timed_block(
    {
        graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_0);
        std::atomic<double> cost_all{0.};
        graph_.apply_all_unary_parallel([&cost_all](auto *n) {
            cost_all += n->dense_->cost_;
        });

        fmt::print("initial cost_total: {}\n", cost_all);
    }
    // );
    for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
        fmt::print("------------------------------------\n");
        fmt::print("Iteration: {}\n", i_iter);
        timed_block_labeled("all",

        // timed_block(
        graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_1);
        // );
        // timed_block(
        graph_.apply_all_binary_forward<true>(ns_riccati::pre_solving_steps_2);
        // );
        // timed_block(
        graph_.apply_all_binary_backward<true>(ns_riccati::backward_pass);
        // );
        // timed_block(
        graph_.apply_all_unary_parallel(ns_riccati::post_solving_steps);
        // );
        // timed_block(
        graph_.apply_all_binary_forward<false, true>(ns_riccati::forward_rollout);
        // );
        // timed_block(
        graph_.apply_all_unary_parallel(ns_riccati::post_rollout_steps);
        // );
        // timed_block(
        graph_.apply_all_unary_parallel(ns_riccati::pre_solving_steps_0);
        // );
        );
        std::atomic<double> cost_all{0.};
        graph_.apply_all_unary_parallel([&cost_all](auto *n) {
            // fmt::print("v:{}\n", n->data_->dense_->cost_);
            cost_all += n->dense_->cost_;
        });

        fmt::print("cost_total: {}\n", cost_all);
        /// @todo: check langrangian
    }
    // });
    moto::utils::timing_storage<"all">::get().count = n_iter;
}
} // namespace moto
