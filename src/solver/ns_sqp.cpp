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
        fmt::print("initial cost_total: {}\n", cost_all.load());
        graph_.apply_all_unary_parallel(nullsp_kkt_solve::update_approx);
    }
    for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
        solver::line_search_cfg ls_config[get_num_threads()];
        solver::line_search_cfg ls_final_cfg;
        auto finalize_bound = [&]() {
            for (size_t i = 0; i < get_num_threads(); ++i) {
                ls_final_cfg.primal.merge_from(ls_config[i].primal);
                ls_final_cfg.dual.merge_from(ls_config[i].dual);
            }
            ls_final_cfg.alpha_primal = ls_final_cfg.primal.alpha_max;
            ls_final_cfg.alpha_dual = ls_final_cfg.dual.alpha_max;
            fmt::print("\talpha_pr:\t{}\n", ls_final_cfg.alpha_primal);
            fmt::print("\talpha_du:\t{}\n", ls_final_cfg.alpha_dual);
        };
        fmt::print("------------------------------------\n");
        fmt::print("SQP Iteration no. {}\n", i_iter);
        timed_block_labeled("all",
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::ns_factorization);
                            graph_.apply_all_binary_forward<true>(nullsp_kkt_solve::partial_value_derivative);
                            graph_.apply_all_binary_backward<true>(nullsp_kkt_solve::riccati_recursion);
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::compute_primal_sensitivity);
                            graph_.apply_all_binary_forward<false, true>(nullsp_kkt_solve::fwd_linear_rollout);
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::finalize_newton_step);
                            graph_.apply_all_unary_parallel([&ls_config](size_t tid, auto *d) {
                                ineq_soft_solve::calculate_line_search_bounds(d, ls_config[tid]);
                            });
                            finalize_bound();
                            graph_.apply_all_unary_parallel([&ls_final_cfg](auto *d) { nullsp_kkt_solve::line_search_step(d, &ls_final_cfg); });
                            graph_.apply_all_unary_parallel(nullsp_kkt_solve::update_approx););
        kkt_info info;
        for (auto &n : graph_.get_unordered_flattened_nodes()) {
            info.objective += n->objective();
            info.inf_prim_res = std::max(n->inf_prim_res(), info.inf_prim_res);
            info.inf_dual_res = std::max(info.inf_dual_res, n->dense_->jac_[__u].cwiseAbs().maxCoeff());
        }
        graph_.apply_all_binary_forward<false, true>([&info](node_data *cur, node_data *next) {
            if (next != nullptr) [[likely]] {
                // cancellation of jacobian from y to x
                static row_vector tmp;
                tmp.conservativeResize(next->dense_->jac_[__x].cols());
                tmp.noalias() = next->dense_->jac_[__x] * permutation_from_y_to_x(cur->ocp_, next->ocp_) + cur->dense_->jac_[__y];
                info.inf_dual_res = std::max(info.inf_dual_res, tmp.cwiseAbs().maxCoeff());
            } else /// @todo: include initial jac[__x] inf norm if init is optimized
                info.inf_dual_res = std::max(info.inf_dual_res, cur->dense_->jac_[__y].cwiseAbs().maxCoeff());
        });
        fmt::print("\tobjective:\t{}\n", info.objective);
        fmt::print("\tinf_prim:\t{}\n", info.inf_prim_res);
        fmt::print("\tinf_dual:\t{}\n", info.inf_dual_res);
        /// @todo: check langrangian
    }
    // });
    moto::utils::timing_storage<"all">::get().count = n_iter;
}
} // namespace moto
