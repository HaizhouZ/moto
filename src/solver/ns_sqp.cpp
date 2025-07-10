#include <moto/solver/ineq_soft_solve.hpp>
#include <moto/solver/ns_riccati_data.hpp>
#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/ns_sqp.hpp>

#define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

namespace moto {
void ns_sqp::forward() {
    timed_block(graph_.apply_all_unary_parallel(nullsp_kkt_solve::update_approx));
}
void ns_sqp::update(size_t n_iter) {

    // timed_block_labeled("all", {
    // timed_block(
    { // rollout for initialization
        fmt::print("Initialization for SQP...\n");
        graph_.apply_all_unary_parallel([](solver::data_base *cur) {
            cur->update_approximation(true);
            ineq_soft_solve::initialize(cur);
        });
        std::atomic<double> cost_all{0.};
        graph_.apply_all_unary_parallel([&cost_all](auto *n) {
            cost_all += n->dense_->cost_;
        });
        fmt::print("initial cost_total: {}\n", cost_all);
        graph_.apply_all_unary_parallel(nullsp_kkt_solve::update_approx);
    }
    constexpr scalar_t a = 1.0; // line search step size
    // );
    for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
        fmt::print("------------------------------------\n");
        fmt::print("Iteration: {}\n", i_iter);
        timed_block_labeled("all",

                            // timed_block(
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::ns_factorization);
                            // );
                            // timed_block(
                            graph_.apply_all_binary_forward<true>(nullsp_kkt_solve::partial_value_derivative);
                            // );
                            // timed_block(
                            graph_.apply_all_binary_backward<true>(nullsp_kkt_solve::riccati_recursion);
                            // );
                            // timed_block(
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::compute_primal_sensitivity);
                            // );
                            // timed_block(
                            graph_.apply_all_binary_forward<false, true>(nullsp_kkt_solve::fwd_linear_rollout);
                            // );
                            // timed_block(
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::finalize_newton_step);
                            // );
                            // timed_block(
                            graph_.apply_all_unary_parallel([a](auto *d) { nullsp_kkt_solve::line_search_step(d, a); });
                            // );
                            // timed_block(
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::update_approx);
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
