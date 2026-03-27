#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/field_conversion.hpp>
#include <numeric>

// #define ENABLE_TIMED_BLOCK
#define SHOW_DETAIL_TIMING
#include <moto/utils/timed_block.hpp>

namespace moto {
ns_sqp::kkt_info ns_sqp::initialize() {
    if (settings.verbose)
        fmt::print("Initialization for SQP...\n");
    if (!settings.ipm.warm_start)
        settings.ipm.mu = settings.ipm.mu0; // initialize mu before setting up workspace data, as it may be used in the workspace data setup
    graph_.for_each_parallel([this](data *cur) {
        // setup solver settings
        cur->for_each_constr([this](const generic_func &c, func_approx_data &d) { c.setup_workspace_data(d, &settings); });
        cur->update_approximation(node_data::update_mode::eval_val);
        // initialize the data
        if (!settings.ipm.warm_start)
            solver::ineq_soft::initialize(cur);
    });
    graph_.for_each_parallel([](data *cur) {
        cur->update_approximation(node_data::update_mode::eval_derivatives);
    });
    kkt_info kkt = compute_kkt_info();
    // print statistics header
    if (settings.verbose) {
        // print_scaling_info();
        print_stat_header();
        print_stats(-1, kkt, false); // print initial stats
    }
    return kkt;
}

void ns_sqp::post_factorization_correction_step() {
    detail_timed_block_start("riccati_recursion_correction");
    graph_.apply_backward(solver_call(&solver_type::riccati_recursion_correction), true);
    detail_timed_block_end("riccati_recursion_correction");
    graph_.for_each_parallel(solver_call(&solver_type::compute_primal_sensitivity_correction));
    graph_.apply_forward(solver_call(&solver_type::fwd_linear_rollout_correction), true);
}
void ns_sqp::finalize_correction(data *d) {
    riccati_solver_->finalize_primal_step_correction(d);
    solver::ineq_soft::finalize_newton_step(d);
}

void ns_sqp::ineq_constr_correction() {
    if (settings.ipm.ipm_enable_affine_step()) {
        graph_.for_each_parallel([this](size_t tid, data *d) {
            solver::ineq_soft::finalize_predictor_step(d, &setting_per_thread[tid]);
        });
        settings.ipm.ipm_end_predictor_computation(); // ipm affine step computation is done
        // collect worker ipm data
        solver::ipm_config::worker &main_worker = setting_per_thread[0];
        for (size_t i : range(settings.n_worker)) {
            main_worker += setting_per_thread[i];
        }
        // adaptive mu update
        settings.ipm.adaptive_mu_update(main_worker);
        // use the new mu to update the rhs cost jacobian
        graph_.for_each_parallel(solver::ineq_soft::corrector_step_start);
        // solve the problem again with updated mu
        post_factorization_correction_step();
        graph_.for_each_parallel([this](data *d) {
            solver::ineq_soft::corrector_step_end(d);
            finalize_correction(d);
        });
        // recompute line search bounds with the corrected newton step
        settings.ls.reset();
        for (solver::linesearch_config &s : setting_per_thread) {
            s.reset();
        }
        graph_.for_each_parallel([this](size_t tid, data *d) {
            solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
        });
        finalize_ls_bound_and_set_to_max();
    }
}
void ns_sqp::ineq_constr_prediction() {
    if (settings.ipm.ipm_enable_affine_step()) { // compute the affine step, no need to finalize dual step
        settings.ipm.ipm_start_predictor_computation();
    }
}
ns_sqp::kkt_info ns_sqp::update(size_t n_iter, bool verbose) {
    settings.verbose = verbose;
    graph_.no_except() = settings.no_except;
    settings.n_worker = graph_.n_jobs();
    kkt_last = initialize();
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// main loop
    try {
        filter_linesearch_data ls;
        //  = {
        //     .points = {{kkt_last.inf_prim_res, kkt_last.inf_dual_res}},
        // };
        ls.constr_vio_min = kkt_last.inf_prim_res * settings.ls.constr_vio_min_frac;
        for ([[maybe_unused]] size_t i_iter : range(n_iter)) {
            bool has_ineq = false;
            // check if there is any inequality constraint using only the key nodes
            for (data &n : graph_.nodes()) {
                if (n.problem().dim(__ineq_x) > 0 || n.problem().dim(__ineq_xu) > 0) {
                    has_ineq = true;
                    break;
                }
            }
            settings.ls.reset();
            setting_per_thread.reset(settings.n_worker);

            timed_block_start("sqp_single_iter");
            detail_timed_block_start("ns factorization");
            graph_.for_each_parallel(solver_call(&solver_type::ns_factorization));
            detail_timed_block_end("ns factorization");

            detail_timed_block_start("riccati_recursion");
            graph_.apply_backward(solver_call(&solver_type::riccati_recursion), true);
            detail_timed_block_end("riccati_recursion");
            detail_timed_block_start("post solve");
            graph_.for_each_parallel(solver_call(&solver_type::compute_primal_sensitivity));
            detail_timed_block_end("post solve");
            detail_timed_block_start("fwd_linear_rollout");
            graph_.apply_forward(solver_call(&solver_type::fwd_linear_rollout), true);
            detail_timed_block_end("fwd_linear_rollout");
            if (has_ineq)
                ineq_constr_prediction();
            detail_timed_block_start("finalize_primal_step");
            graph_.for_each_parallel([this](size_t tid, data *d) {
                riccati_solver_->finalize_primal_step(d);
                solver::ineq_soft::finalize_newton_step(d);
                // decide line search bounds (e.g., fraction-to-bounds)
                solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
            });
            detail_timed_block_end("finalize_primal_step");
            finalize_ls_bound_and_set_to_max();

            detail_timed_block_start("ineq_corrector_step");
            if (has_ineq)
                ineq_constr_correction();
            detail_timed_block_end("ineq_corrector_step");

            // iterative refinement (safe to run without inequality constraints)
            if (has_ineq && settings.rf.enabled && settings.rf.max_iters > 0) {
                iterative_refinement();
            }
            /// @todo: update the line search stepsize?
            // real line search step
            ls.reset_per_iter_data();
            ls.initial_alpha_primal = settings.ls.alpha_primal;
            ls.initial_alpha_dual = settings.ls.alpha_dual;
            ls.best_trial = filter_linesearch_data::trial();
            graph_.for_each_parallel([&](data *d) {
                // finalize dual steps before line search
                riccati_solver_->finalize_dual_newton_step(d);
                d->backup_trial_state();
                solver::ineq_soft::backup_trial_state(d);
            });
        LS_START:
            detail_timed_block_start("apply_affine_step");
            graph_.for_each_parallel([this](data *d) {
                d->restore_trial_state();
                solver::ineq_soft::restore_trial_state(d);
                riccati_solver_->apply_affine_step(d, &settings);
                solver::ineq_soft::apply_affine_step(d, &settings);
                d->update_approximation(node_data::update_mode::eval_val);
            });
            detail_timed_block_end("apply_affine_step");
            detail_timed_block_start("update_res_stat");
            kkt_info kkt_trial = compute_kkt_info(false);
            detail_timed_block_end("update_res_stat");
            const auto update_approx_derivatives = [&] {
                detail_timed_block_start("update_approx_accepted");
                graph_.for_each_parallel([](data *d) {
                    d->update_approximation(node_data::update_mode::eval_derivatives);
                });
                kkt_trial = compute_kkt_info();
                detail_timed_block_end("update_approx_accepted");
            };
            line_search_action action = line_search_action::accept;
            if (settings.ls.enabled) {
                action = filter_linesearch(ls, kkt_trial, kkt_last);
            }
            switch (action) {
            case line_search_action::accept:
                update_approx_derivatives();
                break;
            case line_search_action::retry_second_order_correction:
                second_order_correction();
                goto LS_START;
            case line_search_action::backtrack:
                if (ls.recompute_approx) {
                    // need to recompute the approximation
                    goto LS_START;
                }
                break;
            case line_search_action::stop:
                update_approx_derivatives();
                break;
            }
            timed_block_end("sqp_single_iter");
            // print statistics
            kkt_trial.num_iter = i_iter + 1;
            kkt_trial.ls_steps = ls.step_cnt;
            kkt_last = kkt_trial;
            if (verbose) {
                // print_licq_info();
                print_stats(i_iter, kkt_last, has_ineq);
            }

            if (kkt_last.inf_dual_res < settings.dual_tol &&
                kkt_last.inf_prim_res < settings.prim_tol &&
                kkt_last.inf_comp_res < settings.comp_tol) {
                if (verbose)
                    fmt::print("Converged!\n");
                kkt_last.solved = true;
                break;
            }

            if (has_ineq && settings.ipm.mu_method == solver::ipm_config::monotonic_decrease) {
                bool mu_changed = false;
                while (kkt_last.inf_prim_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_dual_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_comp_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold) {
                    settings.ipm.mu *= settings.ipm.mu_monotone_factor;
                    fmt::print("Monotone decrease of mu: new mu = {:.3e}\n", settings.ipm.mu);
                    ls.points.clear(); // clear the filter to accept the current point
                    mu_changed = true;
                    // ls.points.push_back({kkt_last.inf_prim_res, kkt_last.inf_dual_res});
                }
                if (!mu_changed) {
                    bool prim_fail = kkt_last.inf_prim_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool dual_fail = kkt_last.inf_dual_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool comp_fail = kkt_last.inf_comp_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    fmt::print("Not using monotone decrease of mu: primal res {} threshold, dual res {} threshold, comp res {} threshold\n",
                               prim_fail ? "exceeds" : "within",
                               dual_fail ? "exceeds" : "within",
                               comp_fail ? "exceeds" : "within",
                               settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold);
                }
                settings.ipm.mu = std::max(settings.ipm.mu, 1e-11); // enforce minimum mu
            }

            // });
        }
    } catch (...) {
        if (settings.no_except) {
            if (settings.verbose)
                fmt::print("Exception caught during SQP iterations. Terminating.\n");
        } else {
            throw;
        }
    }
    return kkt_last;
}
ns_sqp::kkt_info ns_sqp::compute_kkt_info(bool update_dual_res) {
    kkt_info kkt;
    field_t max_field;
    vector max_step;
    // scalar_t avg_dual_res = 0;
    for (auto n : graph_.flatten_nodes()) {
        kkt.objective += n->cost();
        kkt.inf_prim_res = std::max(kkt.inf_prim_res, n->inf_prim_res_);
        if (update_dual_res)
            kkt.inf_dual_res = std::max(kkt.inf_dual_res, n->dense().jac_[__u].cwiseAbs().maxCoeff());
        // avg_dual_res += n->dense().jac_[__u].cwiseAbs().maxCoeff();
        kkt.inf_comp_res = std::max(kkt.inf_comp_res, n->inf_comp_res_);
        for (auto f : primal_fields) {
            kkt.inf_prim_step = std::max(kkt.inf_prim_step, n->trial_prim_step[f].cwiseAbs().maxCoeff());
            if (n->trial_prim_step[f].cwiseAbs().maxCoeff() == kkt.inf_prim_step) {
                max_field = f;
                max_step = n->trial_prim_step[f];
            }
            kkt.obj_fullstep_dec += (n->dense().jac_[f].transpose() * n->trial_prim_step[f]).value();
        }
        for (auto f : constr_fields) {
            if (n->trial_dual_step[f].size() > 0) {
                kkt.inf_dual_step = std::max(kkt.inf_dual_step, n->trial_dual_step[f].cwiseAbs().maxCoeff());
            }
        }
    }
    if (update_dual_res) {
        graph_.apply_forward(
            [&](node_data *cur, node_data *next) {
                if (next != nullptr) [[likely]] {
                    // cancellation of jacobian from y to x
                    static row_vector tmp;
                    tmp.conservativeResize(next->dense().jac_[__x].cols());
                    tmp.noalias() = next->dense().jac_[__x] *
                                        utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                                    cur->dense().jac_[__y];
                    kkt.inf_dual_res = std::max(kkt.inf_dual_res, tmp.cwiseAbs().maxCoeff());
                } else { /// @todo: include initial jac[__x] inf norm if init is optimized
                    kkt.inf_dual_res = std::max(kkt.inf_dual_res, cur->dense().jac_[__y].cwiseAbs().maxCoeff());
                }
            },
            true);
    }
    return kkt;
}
void ns_sqp::print_licq_info() {
    // Global LICQ via forward nullspace propagation (DMS staircase structure).
    // Must be called after ns_factorization (nsp_ must be populated).
    //
    // Equality constraints:
    //   A_k = [s_c_stacked_0_K * Z_x  |  s_c_stacked]   (ncstr × (nz_x + nu))
    //   where Z_x spans the null space of all prior-stage constraints in x_k coords.
    //   x_0 is fixed → Z_x starts empty (0 cols).
    //
    // Approximately-active inequality constraints are stacked below the equality rows.
    //   Active criterion: slack[i] < active_tol  (absolute, independent of mu)
    //   Their u-Jacobian and x-Jacobian rows are collected from the ipm_constr approx data.
    //
    // Global LICQ holds iff rank(A_k) == nrows_k at every stage.
    //
    // Null space propagation:  Z_{x,k+1} = [F_x * Z_x | F_u] * null(A_k)
    using namespace solver::ns_riccati;
    constexpr scalar_t active_tol = 1e-3; // slack threshold for approximate activeness

    fmt::print("=== Global LICQ (forward nullspace propagation) ===\n");

    matrix Z_x; // null space of constraints up to stage k, in x_k coords (nx × nz_x)
    bool any_violated = false;
    int stage = 0;

    graph_.apply_forward([&](node_data *cur_nd, node_data *next_nd) {
        auto *d = static_cast<data *>(cur_nd);
        auto &nsp = d->nsp_;
        int nu = (int)d->nu;
        int nx = (int)d->nx;
        int ncstr = (int)d->ncstr; // equality constraints only
        int nz_x = (int)Z_x.cols();

        // Collect approximately-active inequality constraint Jacobians.
        //
        // __ineq_xu:  direct Jacobians w.r.t. __x and __u.
        //
        // __ineq_x:   x-arg is substituted to __y (= x_{k+1}) in finalize_impl, so
        //             jac w.r.t. __y gives dg/dy_k.  Map back to (x_k, u_k) coords via dynamics:
        //               u-col = (dg/dy_k) * F_u
        //               x-col = (dg/dy_k) * F_x      (no direct x_k dependence after substitution)
        //             This mirrors how s_c_stacked / s_c_stacked_0_K are built in ns_factorization.
        matrix ineq_jac_u; // (n_active × nu)
        matrix ineq_jac_x; // (n_active × nx)
        int n_active = 0;

        const matrix F_u_dense = d->F_u.dense();
        const matrix F_x_dense = d->F_x.dense();

        const auto collect_ineq_xu = [&](const soft_constr &sf, soft_constr::approx_data &sd) {
            const auto *id = dynamic_cast<const solver::ipm_constr::approx_data *>(&sd);
            if (!id)
                return;
            int m = (int)sf.dim();
            for (int i = 0; i < m; ++i) {
                if (id->slack_(i) >= active_tol)
                    continue;
                ineq_jac_u.conservativeResize(n_active + 1, nu);
                ineq_jac_x.conservativeResize(n_active + 1, nx);
                ineq_jac_u.row(n_active).setZero();
                ineq_jac_x.row(n_active).setZero();
                size_t arg_idx = 0;
                for (const sym &arg : sf.in_args()) {
                    const auto &J = id->jac_[arg_idx];
                    if (J.rows() > 0) {
                        if (arg.field() == __u)
                            ineq_jac_u.row(n_active) = J.row(i);
                        else if (arg.field() == __x)
                            ineq_jac_x.row(n_active) = J.row(i);
                    }
                    arg_idx++;
                }
                n_active++;
            }
        };
        const auto collect_ineq_x = [&](const soft_constr &sf, soft_constr::approx_data &sd) {
            // __ineq_x: jac is w.r.t. __y (= x_{k+1}), map through dynamics
            const auto *id = dynamic_cast<const solver::ipm_constr::approx_data *>(&sd);
            if (!id)
                return;
            int m = (int)sf.dim();
            for (int i = 0; i < m; ++i) {
                if (id->slack_(i) >= active_tol)
                    continue;
                ineq_jac_u.conservativeResize(n_active + 1, nu);
                ineq_jac_x.conservativeResize(n_active + 1, nx);
                ineq_jac_u.row(n_active).setZero();
                ineq_jac_x.row(n_active).setZero();
                size_t arg_idx = 0;
                for (const sym &arg : sf.in_args()) {
                    const auto &J = id->jac_[arg_idx];
                    if (J.rows() > 0)
                        if (arg.field() == __y) {
                            // dg/dy_k: project through F_u (u-col) and F_x (x-col)
                            ineq_jac_u.row(n_active).noalias() += J.row(i) * F_u_dense;
                            ineq_jac_x.row(n_active).noalias() += J.row(i) * F_x_dense;
                        } else if (arg.field() == __x) {
                            // direct x_k dependence (if any) after substitution
                            ineq_jac_x.row(n_active).noalias() += J.row(i);
                        }
                    arg_idx++;
                }
                n_active++;
            }
        };
        d->template for_each(__ineq_xu, collect_ineq_xu);
        d->template for_each(__ineq_x, collect_ineq_x);

        int nrows = ncstr + n_active;

        int rank_Ak;
        matrix Z_k; // null(A_k) in [t_x; t_u] coords, shape (nz_x + nu) × nz_new

        if (nrows == 0) {
            rank_Ak = 0;
            Z_k = matrix::Identity(nz_x + nu, nz_x + nu);

        } else if (nz_x == 0 && n_active == 0) {
            // Pure equality, first stage: reuse ns_factorization result
            rank_Ak = (int)nsp.rank;
            if (d->rank_status_ == fully_constrained) {
                Z_k = matrix::Zero(nu, 0);
            } else if (d->rank_status_ == constrained) {
                Z_k = nsp.Z_u;
            } else {
                Z_k = matrix::Identity(nu, nu);
            }

        } else {
            // General case: form augmented A_k = [eq_rows ; active_ineq_rows]
            matrix A_k(nrows, nz_x + nu);
            A_k.setZero();
            if (ncstr > 0) {
                if (nz_x > 0)
                    A_k.topLeftCorner(ncstr, nz_x).noalias() = nsp.s_c_stacked_0_K * Z_x;
                A_k.topRightCorner(ncstr, nu) = nsp.s_c_stacked;
            }
            if (n_active > 0) {
                if (nz_x > 0)
                    A_k.bottomLeftCorner(n_active, nz_x).noalias() = ineq_jac_x * Z_x;
                A_k.bottomRightCorner(n_active, nu) = ineq_jac_u;
            }
            Eigen::FullPivLU<matrix> lu_Ak(A_k);
            rank_Ak = (int)lu_Ak.rank();
            Z_k = lu_Ak.kernel();
        }

        bool stage_licq = (rank_Ak == nrows);
        if (!stage_licq)
            any_violated = true;

        fmt::print("  stage {:d}  neq {:d}  nineq_active {:d}  nu {:d}  nz_x {:d} → nz {:d}  rank {:d}/{:d}{:s}\n",
                   stage, ncstr, n_active, nu, nz_x, (int)Z_k.cols(), rank_Ak, nrows,
                   stage_licq ? "" : "  *** GLOBAL LICQ VIOLATED ***");

        // Propagate null space: Z_{x,k+1} = [F_x * Z_x | F_u] * Z_k
        if (next_nd != nullptr) {
            int ny = (int)d->ny;
            matrix FZ(ny, nz_x + nu);
            FZ.setZero();
            if (nz_x > 0)
                FZ.leftCols(nz_x).noalias() = d->F_x.dense() * Z_x;
            FZ.rightCols(nu) = d->F_u.dense();
            Z_x.noalias() = FZ * Z_k;
        }
        ++stage;
    },
                         true);

    fmt::print("  Global LICQ: {:s}\n", any_violated ? "*** VIOLATED ***" : "OK");
    fmt::print("===================================================\n");
}
void ns_sqp::print_scaling_info() {
    fmt::print("=== Scaling info ===\n");
    for (auto n : graph_.flatten_nodes()) {
        fmt::print("  Node:\n");
        for (auto cf : constr_fields) {
            const auto &approx = n->dense().approx_[cf];
            if (approx.v_.size() == 0)
                continue;
            scalar_t res_inf = approx.v_.cwiseAbs().maxCoeff();
            fmt::print("    constr {:12s}  res inf-norm: {:.3e}", field::name(cf), res_inf);
            for (auto pf : primal_fields) {
                const auto &jac = approx.jac_[pf];
                if (jac.is_empty())
                    continue;
                matrix jac_dense = jac.dense();
                scalar_t jac_inf = jac_dense.cwiseAbs().maxCoeff();
                fmt::print("  |J_{:s}| inf: {:.3e}", field::name(pf), jac_inf);
            }
            fmt::print("\n");
        }
        // cost Jacobians
        fmt::print("    cost");
        for (auto pf : primal_fields) {
            const auto &jac = n->dense().jac_[pf];
            if (jac.size() == 0)
                continue;
            scalar_t jac_inf = jac.cwiseAbs().maxCoeff();
            fmt::print("  |J_{:s}| inf: {:.3e}", field::name(pf), jac_inf);
        }
        fmt::print("\n");
    }
    fmt::print("====================\n");
}
} // namespace moto
