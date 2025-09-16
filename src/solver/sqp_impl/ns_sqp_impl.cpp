#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/field_conversion.hpp>
#include <numeric>

// #define ENABLE_TIMED_BLOCK
#include <moto/utils/timed_block.hpp>

#define SHOW_DETAIL_TIMING

namespace moto {
ns_sqp::kkt_info ns_sqp::initialize() {
    if (settings.verbose)
        fmt::print("Initialization for SQP...\n");
    graph_.for_each_parallel([this](data *cur) {
        // setup solver settings
        cur->for_each_constr([this](const generic_func &c, func_approx_data &d) { c.setup_workspace_data(d, &settings); });
        cur->update_approximation(node_data::update_mode::eval_val);
        // initialize the data
        solver::ineq_soft::initialize(cur);
    });
    graph_.for_each_parallel([](data *cur) {
        cur->update_approximation(node_data::update_mode::eval_derivatives);
    });
    kkt_info kkt = compute_kkt_info();
    // print statistics header
    if (settings.verbose) {
        print_stat_header();
        print_stats(-1, kkt, false); // print initial stats
    }
    return kkt;
}

void ns_sqp::correction_step() {
    detail_timed_block_start("riccati_recursion_correction");
    graph_.apply_backward(bind(&solver_type::riccati_recursion_correction), true);
    detail_timed_block_end("riccati_recursion_correction");
    graph_.for_each_parallel(bind(&solver_type::compute_primal_sensitivity_correction));
    graph_.apply_forward(bind(&solver_type::fwd_linear_rollout_correction), true);
}
void ns_sqp::finalize_correction(data *d) {
    riccati_solver_->finalize_newton_step_correction(d);
    solver::ineq_soft::finalize_newton_step(d);
    riccati_solver_->finalize_dual_newton_step(d);
}

ns_sqp::kkt_info ns_sqp::update(size_t n_iter, bool verbose) {
    settings.verbose = verbose;
    settings.n_worker = graph_.n_jobs();
    kkt_last = initialize();
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// main loop
    try {
        for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
            bool has_ineq = false;
            // check if there is any inequality constraint using only the key nodes
            for (data &n : graph_.nodes()) {
                if (n.problem().dim(__ineq_x) > 0 || n.problem().dim(__ineq_xu) > 0) {
                    has_ineq = true;
                    break;
                }
            }
            settings.as<solver::linesearch_config>().reset();
            setting_per_thread.reset(settings.n_worker);

            timed_block_start("sqp_single_iter");
            detail_timed_block_start("ns factorization");
            graph_.for_each_parallel(bind(&solver_type::ns_factorization));
            detail_timed_block_end("ns factorization");

            detail_timed_block_start("riccati_recursion");
            graph_.apply_backward(bind(&solver_type::riccati_recursion), true);
            detail_timed_block_end("riccati_recursion");
            detail_timed_block_start("post solve");
            graph_.for_each_parallel(bind(&solver_type::compute_primal_sensitivity));
            detail_timed_block_end("post solve");
            detail_timed_block_start("fwd_linear_rollout");
            graph_.apply_forward(bind(&solver_type::fwd_linear_rollout), true);
            detail_timed_block_end("fwd_linear_rollout");

            bool finalize_dual = true;

            if (has_ineq && settings.ipm_enable_affine_step()) { // compute the affine step, no need to finalize dual step
                settings.ipm_start_predictor_computation();
                finalize_dual = false; // do not finalize dual step
            }
            detail_timed_block_start("finalize_newton_step");
            graph_.for_each_parallel([finalize_dual, this](size_t tid, data *d) {
                riccati_solver_->finalize_newton_step(d, finalize_dual);
                solver::ineq_soft::finalize_newton_step(d);
                // decide line search bounds (e.g., fraction-to-bounds)
                solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
            });
            detail_timed_block_end("finalize_newton_step");
            detail_timed_block_start("corrector_step");
            finalize_ls_bound_and_set_to_max();
            if (has_ineq && settings.ipm_enable_affine_step()) {
                // line search with max bounds
                graph_.for_each_parallel([this](size_t tid, data *d) {
                    solver::ineq_soft::finalize_predictor_step(d, &setting_per_thread[tid]);
                });
                settings.ipm_end_predictor_computation(); // ipm affine step computation is done
                // collect worker ipm data
                solver::ipm_config::worker &main_worker = setting_per_thread[0];
                for (size_t i : range(settings.n_worker)) {
                    main_worker += setting_per_thread[i];
                }
                // adaptive mu update
                settings.adaptive_mu_update(main_worker);
                // use the new mu to update the rhs jacobian
                graph_.for_each_parallel(solver::ineq_soft::corrector_step_start);
                // solve the problem again with updated mu
                correction_step();
                graph_.for_each_parallel([this](data *d) {
                    solver::ineq_soft::corrector_step_end(d);
                    finalize_correction(d);
                });
                // recompute line search bounds with the corrected newton step
                settings.as<solver::linesearch_config>().reset();
                for (solver::linesearch_config &s : setting_per_thread) {
                    s.reset();
                }
                graph_.for_each_parallel([this](size_t tid, data *d) {
                    solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
                });
                finalize_ls_bound_and_set_to_max();
            }
            detail_timed_block_end("corrector_step");
            // iterative refinement
            if (has_ineq && settings.use_iterative_refinement) {
                iterative_refinement();
            }
            /// @todo: update the line search stepsize?
            // real line search step
            ls_info ls;
            ls.initial_alpha_primal = settings.alpha_primal;
            ls.initial_alpha_dual = settings.alpha_dual;
        LS_START:
            detail_timed_block_start("line_search_step");
            graph_.for_each_parallel([this](data *d) {
                riccati_solver_->apply_affine_step(d, &settings);
                solver::ineq_soft::apply_affine_step(d, &settings);
            });
            detail_timed_block_end("line_search_step");
            detail_timed_block_start("update_approx");
            graph_.for_each_parallel(data::update_approx);
            detail_timed_block_end("update_approx");
            detail_timed_block_start("update_res_stat");
            kkt_info kkt_trial = compute_kkt_info();
            detail_timed_block_end("update_res_stat");
            // basic globalization
            if (settings.use_line_search) {
                backtrack_linesearch(ls, kkt_trial);
                if (ls.recompute_approx) {
                    // need to recompute the approximation
                    goto LS_START;
                }
            }
            timed_block_end("sqp_single_iter");
            // print statistics
            kkt_trial.num_iter = i_iter + 1;
            kkt_trial.ls_steps = ls.step_cnt;
            kkt_last = kkt_trial;
            if (verbose)
                print_stats(i_iter, kkt_last, has_ineq);

            if (kkt_last.inf_dual_res < settings.dual_tol &&
                kkt_last.inf_prim_res < settings.prim_tol &&
                kkt_last.inf_comp_res < settings.comp_tol) {
                if (verbose)
                    fmt::print("Converged!\n");
                kkt_last.solved = true;
                break;
            }
            // });
        }
    } catch (const std::exception &e) {
        // if (verbose)
        fmt::print("Exception caught during SQP iterations: {}\n", e.what());
    }
    return kkt_last;
}
ns_sqp::kkt_info ns_sqp::compute_kkt_info() {
    kkt_info kkt;
    field_t max_field;
    vector max_step;
    // scalar_t avg_dual_res = 0;
    for (auto n : graph_.flatten_nodes()) {
        kkt.objective += n->cost();
        kkt.inf_prim_res = std::max(kkt.inf_prim_res, n->inf_prim_res_);
        kkt.inf_dual_res = std::max(kkt.inf_dual_res, n->dense().jac_[__u].cwiseAbs().maxCoeff());
        // avg_dual_res += n->dense().jac_[__u].cwiseAbs().maxCoeff();
        kkt.inf_comp_res = std::max(kkt.inf_comp_res, n->inf_comp_res_);
        for (auto f : primal_fields) {
            kkt.inf_prim_step = std::max(kkt.inf_prim_step, n->prim_step[f].cwiseAbs().maxCoeff());
            if (n->prim_step[f].cwiseAbs().maxCoeff() == kkt.inf_prim_step) {
                max_field = f;
                max_step = n->prim_step[f];
            }
        }
        for (auto f : constr_fields) {
            if (n->dual_step[f].size() > 0) {
                kkt.inf_dual_step = std::max(kkt.inf_dual_step, n->dual_step[f].cwiseAbs().maxCoeff());
            }
        }
    }
    // avg_dual_res /= graph_.flatten_nodes().size();
    // // avg_dual_res = 0;
    size_t step = 0;
    size_t idx = 0;
    graph_.apply_forward(
        [&](node_data *cur, node_data *next) {
            if (next != nullptr) [[likely]] {
                // cancellation of jacobian from y to x
                static row_vector tmp;
                tmp.conservativeResize(next->dense().jac_[__x].cols());
                tmp.noalias() = next->dense().jac_[__x] *
                                    utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                                cur->dense().jac_[__y];
                // kkt.inf_dual_res = std::max(kkt.inf_dual_res, tmp.cwiseAbs().maxCoeff());
                if (kkt.inf_dual_res < tmp.cwiseAbs().maxCoeff()) {
                    kkt.inf_dual_res = tmp.cwiseAbs().maxCoeff();
                    idx = step;
                }
                if (step == 69) {
                    // fmt::println("------ step {} dual_res: ", step);
                    // fmt::println("y dual res {}", tmp);
                    // fmt::println("y dual res {}", next->dense().jac_[__x] *
                    // utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                    // cur->dense().jac_[__y]);
                }
                // avg_dual_res += tmp.cwiseAbs().maxCoeff();
            } else { /// @todo: include initial jac[__x] inf norm if init is optimized
                // kkt.inf_dual_res = std::max(kkt.inf_dual_res, cur->dense().jac_[__y].cwiseAbs().maxCoeff());
                // avg_dual_res += cur->dense().jac_[__y].cwiseAbs().maxCoeff();
                if (kkt.inf_dual_res < cur->dense().jac_[__y].cwiseAbs().maxCoeff()) {
                    kkt.inf_dual_res = cur->dense().jac_[__y].cwiseAbs().maxCoeff();
                    idx = step;
                }
            }
            step++;
            // // fmt::println("prim {}: {}", step, cur->prim_step[__x
            // // fmt::println("prim {}: {}", step, cur->value(__u).transpose());
            // for (auto f : constr_fields) {
            //     if (cur->dense().dual_[f].size() > 0)
            //         fmt::println("dual {}: {}", f, cur->dense().dual_[f].transpose());
            // }
            // fmt::println("dual {}: {}", step, cur->dense().dual_[__ineq_xu].transpose());
            // fmt::println("jac  {}: {}", step++, cur->dense().jac_[__u]);
            // fmt::println("{}", cur->dense().jac_[__y].cwiseAbs().maxCoeff());
        },
        true);
    // avg_dual_res /= graph_.flatten_nodes().size();
    // fmt::print("max dual res at step {}\n", idx);
    return kkt;
}
} // namespace moto
