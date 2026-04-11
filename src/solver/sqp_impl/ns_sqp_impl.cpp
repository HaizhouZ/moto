#include <magic_enum/magic_enum.hpp>
#include <moto/ocp/ineq_constr.hpp>
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
namespace {

template <typename T>
void reset_part(T &part) {
    part.~T();
    new (&part) T{};
}

template <typename enum_type>
constexpr bool has_flag(enum_type mask, enum_type flag) {
    using mask_t = std::underlying_type_t<enum_type>;
    return (static_cast<mask_t>(mask) & static_cast<mask_t>(flag)) != 0;
}

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

scalar_t max_abs_or_zero(const row_vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}
} // namespace

void ns_sqp::profile_state::reset() {
    total_ms.fill(0.0);
    calls.fill(0);
    iter_ms.fill(0.0);
    iter_calls.fill(0);
    update_start = {};
    iter_start = {};
    current_trial_evaluations = 0;
    total_trial_evaluations = 0;
    iterations.clear();
}

void ns_sqp::profile_state::start_update() {
    reset();
    update_start = profile_clock::now();
}

void ns_sqp::profile_state::finish_update(profile_report &report) const {
    report = {};
    report.total_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - update_start).count();
    report.initialize_ms = total_ms[static_cast<size_t>(profile_phase::initialize_total)];
    report.sqp_iterations = iterations.size();
    report.trial_evaluations = total_trial_evaluations;
    report.iterations = iterations;
    report.phases.reserve(static_cast<size_t>(profile_phase::count));
    for (size_t i = 0; i < static_cast<size_t>(profile_phase::count); ++i) {
        if (calls[i] == 0) {
            continue;
        }
        report.phases.push_back(profile_phase_stat{
            .name = profile_phase_name(static_cast<profile_phase>(i)),
            .total_ms = total_ms[i],
            .avg_ms = total_ms[i] / static_cast<double>(calls[i]),
            .calls = calls[i],
            .share_of_update = report.total_ms > 0.0 ? total_ms[i] / report.total_ms : 0.0,
        });
    }
}

void ns_sqp::profile_state::start_iteration(size_t index) {
    iter_ms.fill(0.0);
    iter_calls.fill(0);
    current_trial_evaluations = 0;
    iter_start = profile_clock::now();
    iterations.push_back(profile_iteration{.index = index + 1});
}

void ns_sqp::profile_state::finish_iteration(size_t ls_steps) {
    auto &iter = iterations.back();
    iter.total_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - iter_start).count();
    iter.ls_steps = ls_steps;
    iter.trial_evaluations = current_trial_evaluations;
}

void ns_sqp::profile_state::record(profile_phase phase, double elapsed_ms) {
    const size_t idx = static_cast<size_t>(phase);
    total_ms[idx] += elapsed_ms;
    ++calls[idx];
    iter_ms[idx] += elapsed_ms;
    ++iter_calls[idx];
}

void ns_sqp::profile_state::bump_trial_evaluation() {
    ++current_trial_evaluations;
    ++total_trial_evaluations;
}

void ns_sqp::data::backup_trial_state() {
    ns_riccati_data::backup_trial_state();
    solver::ineq_soft::backup_trial_state(this);
}

void ns_sqp::data::restore_trial_state() {
    ns_riccati_data::restore_trial_state();
    solver::ineq_soft::restore_trial_state(this);
}

ns_sqp::scoped_profile::scoped_profile(ns_sqp *owner_, profile_phase phase_)
    : owner(owner_), phase(phase_), start(profile_clock::now()) {}

ns_sqp::scoped_profile::scoped_profile(scoped_profile &&rhs) noexcept
    : owner(rhs.owner), phase(rhs.phase), start(rhs.start) {
    rhs.owner = nullptr;
}

ns_sqp::scoped_profile::~scoped_profile() {
    if (owner == nullptr) {
        return;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - start).count();
    owner->profiler_.record(phase, elapsed_ms);
}

const char *ns_sqp::profile_phase_name(profile_phase phase) {
    auto name = magic_enum::enum_name(phase);
    return name.empty() ? "unknown" : name.data();
}

void ns_sqp::reset_profile() {
    profiler_.reset();
    profile_report_ = {};
}

void ns_sqp::refresh_problem_flags() {
    settings.has_ineq_soft = false;
    settings.has_ipm_ineq = false;
    for (data &n : active_data().nodes()) {
        for (auto ct : ineq_soft_constr_fields) {
            if (n.problem().dim(ct) > 0) {
                settings.has_ineq_soft = true;
                break;
            }
        }
        for (auto ct : ineq_constr_fields) {
            if (n.problem().dim(ct) > 0) {
                settings.has_ipm_ineq = true;
                break;
            }
        }
        if (settings.has_ineq_soft && settings.has_ipm_ineq) {
            break;
        }
    }
}

ns_sqp::kkt_info ns_sqp::initialize() {
    auto total_profile = profile_scope(profile_phase::initialize_total);
    auto &graph = active_data();
    if (settings.verbose)
        fmt::print("Initialization for SQP...\n");
    refresh_problem_flags();
    if (!settings.ipm.warm_start)
        settings.ipm.mu = settings.ipm.mu0; // initialize mu before setting up workspace data, as it may be used in the workspace data setup
    {
        auto phase_profile = profile_scope(profile_phase::initialize_setup_eval);
        graph.for_each_parallel([this](data *cur) {
            // setup solver settings
            cur->for_each_constr([this](const generic_func &c, func_approx_data &d) { c.setup_workspace_data(d, &settings); });
            solver::ineq_soft::bind_runtime(cur);
            cur->update_approximation(node_data::update_mode::eval_all);
        });
        settings.initialized = true;
    }
    initialize_equality_multipliers();
    kkt_info kkt;
    {
        auto phase_profile = profile_scope(profile_phase::initialize_kkt);
        update_primal_info(kkt, point_value_mask::primal | point_value_mask::barrier_objective);
        update_stat_info(kkt);
    }
    reset_scaling(); // clear scale vectors; will be recomputed on first iteration
    // print statistics header
    if (settings.verbose) {
        // print_scaling_info();
        print_stat_header();
        print_stats(kkt, iter_info{}, 0); // print initial stats
    }
    return kkt;
}
void ns_sqp::post_factorization_correction_step() {
    auto total_profile = profile_scope(profile_phase::correction_post_factorization);
    auto &graph = active_data();
    {
        auto phase_profile = profile_scope(profile_phase::correction_riccati_recursion);
        detail_timed_block_start("riccati_recursion_correction");
        graph.apply_backward(solver_call(&solver_type::riccati_recursion_correction), true);
        detail_timed_block_end("riccati_recursion_correction");
    }
    {
        auto phase_profile = profile_scope(profile_phase::correction_primal_sensitivity);
        graph.for_each_parallel(solver_call(&solver_type::compute_primal_sensitivity_correction));
    }
    {
        auto phase_profile = profile_scope(profile_phase::correction_fwd_rollout);
        graph.apply_forward(solver_call(&solver_type::fwd_linear_rollout_correction), true);
    }
}
void ns_sqp::finalize_correction(data *d) {
    riccati_solver_->finalize_primal_step_correction(d);
    if (use_normal_soft_phase() || in_restoration_phase()) {
        solver::ineq_soft::finalize_newton_step(d);
    }
}

void ns_sqp::reset_ls_workers() {
    settings.ls.reset();
    setting_per_thread.reset(settings.n_worker);
}

void ns_sqp::refresh_ls_bounds() {
    reset_ls_workers();
    auto &graph = active_data();
    graph.for_each_parallel([this](size_t tid, data *d) {
        if (use_normal_soft_phase() || in_restoration_phase()) {
            solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
        }
    });
    finalize_ls_bound_and_set_to_max();
}

void ns_sqp::ineq_constr_correction(iteration_context &ctx) {
    if (settings.ipm.ipm_enable_affine_step()) {
        auto &graph = active_data();
        for (auto &worker_cfg : setting_per_thread) {
            worker_cfg.as<solver::ipm_config::worker_type>() = {};
        }
        graph.for_each_parallel([this](size_t tid, data *d) {
            solver::ineq_soft::finalize_predictor_step(d, &setting_per_thread[tid]);
        });
        settings.ipm.ipm_end_predictor_computation(); // ipm affine step computation is done
        // collect worker ipm data
        solver::ipm_config::worker main_worker{};
        for (size_t i : range(settings.n_worker)) {
            main_worker += setting_per_thread[i].as<solver::ipm_config::worker_type>();
        }
        if (settings.has_ipm_ineq) {
            settings.ipm.adaptive_mu_update(main_worker);
            settings.ipm.mu = settings.ipm.mu_trial;
            ctx.mu_changed = true;
        }
        run_correction_step(
            solver::ineq_soft::corrector_step_start,
            [this](data *d) {
                solver::ineq_soft::corrector_step_end(d);
                finalize_correction(d);
            });
    }
}
void ns_sqp::ineq_constr_prediction() {
    if (settings.ipm.ipm_enable_affine_step()) { // compute the affine step, no need to finalize dual step
        settings.ipm.ipm_start_predictor_computation();
    }
}

void ns_sqp::solve_direction(iteration_context &ctx, bool do_scaling, bool gauss_newton) {
    auto phase_profile = profile_scope(profile_phase::solve_direction);
    auto &graph = active_data();
    if (do_scaling) {
        auto scaling_profile = profile_scope(profile_phase::scaling);
        detail_timed_block_start("scaling");
        compute_and_apply_scaling(ctx.current);
        detail_timed_block_end("scaling");
    }

    {
        auto ns_factor_profile = profile_scope(profile_phase::ns_factorization);
        detail_timed_block_start("ns factorization");
        graph.for_each_parallel([this, gauss_newton](data *d) {
            riccati_solver_->ns_factorization(d, gauss_newton);
        });
        detail_timed_block_end("ns factorization");
    }

    {
        auto recursion_profile = profile_scope(profile_phase::riccati_recursion);
        detail_timed_block_start("riccati_recursion");
        graph.apply_backward(solver_call(&solver_type::riccati_recursion), true);
        detail_timed_block_end("riccati_recursion");
    }
    {
        auto post_solve_profile = profile_scope(profile_phase::post_solve);
        detail_timed_block_start("post solve");
        graph.for_each_parallel(solver_call(&solver_type::compute_primal_sensitivity));
        detail_timed_block_end("post solve");
    }
    {
        auto rollout_profile = profile_scope(profile_phase::fwd_linear_rollout);
        detail_timed_block_start("fwd_linear_rollout");
        graph.apply_forward(solver_call(&solver_type::fwd_linear_rollout), true);
        detail_timed_block_end("fwd_linear_rollout");
    }

    if (settings.has_ineq_soft)
        ineq_constr_prediction();
    {
        auto finalize_profile = profile_scope(profile_phase::finalize_primal_step);
        detail_timed_block_start("finalize_primal_step");
        graph.for_each_parallel([this](size_t tid, data *d) {
            riccati_solver_->finalize_primal_step(d);
            if (use_normal_soft_phase() || in_restoration_phase()) {
                solver::ineq_soft::finalize_newton_step(d);
                solver::ineq_soft::update_ls_bounds(d, &setting_per_thread[tid]);
            }
        });
        detail_timed_block_end("finalize_primal_step");
    }
    finalize_ls_bound_and_set_to_max();
}

void ns_sqp::correct_direction(iteration_context &ctx, bool do_refinement) {
    auto phase_profile = profile_scope(profile_phase::correct_direction);
    auto &graph = active_data();
    {
        auto corrector_profile = profile_scope(profile_phase::ineq_corrector_step);
        detail_timed_block_start("ineq_corrector_step");
        if (settings.has_ineq_soft)
            ineq_constr_correction(ctx);
        detail_timed_block_end("ineq_corrector_step");
    }

    if (do_refinement && settings.rf.enabled && settings.rf.max_iters > 0)
        iterative_refinement();
}

void ns_sqp::prepare_globalization(filter_linesearch_data &ls, iteration_context &ctx) {
    auto phase_profile = profile_scope(profile_phase::prepare_globalization);
    auto &graph = active_data();
    ls.reset_per_iter_data();
    ls.initial_alpha_primal = settings.ls.alpha_primal;
    ls.initial_alpha_dual = settings.ls.alpha_dual;
    ls.best_trial = filter_linesearch_data::trial();
    ls.merit_fullstep = std::numeric_limits<scalar_t>::infinity();
    ls.best_merit_trial = filter_linesearch_data::merit_trial{};

    graph.for_each_parallel([this](size_t tid, data *d) {
        riccati_solver_->finalize_dual_newton_step(d);
        unscale_duals(d);
        d->backup_trial_state();
    });
    if (ctx.mu_changed) {
        update_primal_info(ctx.current, point_value_mask::barrier_objective);
        ls.points.clear(); // the QP objective changed, so old filter points are no longer comparable
    }
    // Current accepted iterate already has residual information; only refresh
    // line-search quantities for the new Newton direction.
    update_step_info(ctx.current, step_info_mask::orig_step | step_info_mask::barrier_step);
}

bool ns_sqp::evaluate_trial_point(filter_linesearch_data &ls, iteration_context &ctx) {
    auto phase_profile = profile_scope(profile_phase::evaluate_trial_point);
    auto &graph = active_data();
    profiler_.bump_trial_evaluation();
    {
        auto apply_profile = profile_scope(profile_phase::apply_affine_step);
        detail_timed_block_start("apply_affine_step");
        graph.for_each_parallel([this](data *d) {
            d->restore_trial_state();
            riccati_solver_->apply_affine_step(d, &settings);
            if (use_normal_soft_phase() || in_restoration_phase()) {
                solver::ineq_soft::apply_affine_step(d, &settings);
            }
            d->update_approximation(node_data::update_mode::eval_val, true);
        });
        detail_timed_block_end("apply_affine_step");
    }

    {
        auto res_profile = profile_scope(profile_phase::update_res_stat);
        detail_timed_block_start("update_res_stat");
        if (settings.ls.method == linesearch_setting::search_method::merit_backtracking ||
            in_restoration_phase()) {
            graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
            update_stat_info(ctx.trial);
        }
        // compute the primal anyway
        update_primal_info(ctx.trial, point_value_mask::primal | point_value_mask::barrier_objective);
        detail_timed_block_end("update_res_stat");
    }
    return settings.ls.method == linesearch_setting::search_method::merit_backtracking;
}

void ns_sqp::accept_trial_point(filter_linesearch_data &ls, iteration_context &ctx) {
    auto phase_profile = profile_scope(profile_phase::accept_trial_point);
    auto &graph = active_data();
    {
        auto accepted_profile = profile_scope(profile_phase::update_approx_accepted);
        detail_timed_block_start("update_approx_accepted");
        if (settings.ls.method != linesearch_setting::search_method::merit_backtracking) {
            graph.for_each_parallel([this](data *d) {
                d->update_approximation(node_data::update_mode::eval_derivatives, true);
            });
            update_stat_info(ctx.trial);
        }
        detail_timed_block_end("update_approx_accepted");
    }
    ctx.current.accept(ctx.trial);
}

ns_sqp::line_search_action ns_sqp::select_globalization_action(filter_linesearch_data &ls, iteration_context &ctx) {
    if (!settings.ls.enabled || ls.stop) {
        return line_search_action::accept;
    }
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
        return merit_linesearch(ls, ctx.trial, ctx.current);
    }
    return filter_linesearch(ls, ctx.trial, ctx.current);
}

ns_sqp::line_search_action ns_sqp::handle_globalization_failure(filter_linesearch_data &ls, iteration_context &ctx) {
    if (!settings.ls.enabled) {
        accept_trial_point(ls, ctx);
        return line_search_action::accept;
    }

    if (ls.failure_reason == filter_linesearch_per_iter_data::failure_reason_t::tiny_step) {
        auto &graph = active_data();
        graph.for_each_parallel([this](data *d) {
            d->restore_trial_state();
            d->update_approximation(node_data::update_mode::eval_all, true);
        });
        update_primal_info(ctx.current, point_value_mask::primal | point_value_mask::barrier_objective);
        update_stat_info(ctx.current);
        return line_search_action::failure;
    }

    // Line-search fallback strategies select a concrete backup step
    // (best trial or min step). Re-evaluate that selected step and accept
    // it so the control flow matches the line-search decision.
    throw std::runtime_error("Line-search failed for unknown reason");
}

ns_sqp::line_search_action ns_sqp::run_globalization(filter_linesearch_data &ls, iteration_context &ctx) {
    auto phase_profile = profile_scope(profile_phase::run_globalization);
    while (true) {
        bool has_trial_derivatives = evaluate_trial_point(ls, ctx);

        ctx.action = select_globalization_action(ls, ctx);

        switch (ctx.action) {
        case line_search_action::accept:
            if (!has_trial_derivatives)
                accept_trial_point(ls, ctx);
            else
                ctx.current.accept(ctx.trial);
            return ctx.action;
        case line_search_action::retry_second_order_correction:
            second_order_correction();
            break;
        case line_search_action::backtrack:
            if (!ls.recompute_approx)
                throw std::runtime_error("Backtrack requested without recompute");
            break;
        case line_search_action::failure:
            return handle_globalization_failure(ls, ctx);
        }
    }
}

ns_sqp::line_search_action ns_sqp::sqp_iter(filter_linesearch_data &ls, kkt_info &kkt_current,
                                            bool do_scaling, bool do_refinement, bool gauss_newton) {
    auto phase_profile = profile_scope(profile_phase::sqp_iter_total);
    iteration_context ctx{
        .current = kkt_current, // must do this because prepare_globalization will only update the step info
        .phase = in_restoration_phase() ? iteration_phase::restoration : iteration_phase::normal,
    };
    reset_ls_workers();
    solve_direction(ctx, do_scaling, gauss_newton);
    correct_direction(ctx, do_refinement);
    kkt_info current_backup;
    prepare_globalization(ls, ctx);
    current_backup = ctx.current;
    line_search_action action = run_globalization(ls, ctx);
    if (action == line_search_action::failure)
        kkt_current = current_backup;
    else
        kkt_current = ctx.current;
    return action;
}

void ns_sqp::update_step_info(kkt_info &kkt, step_info_mask mask) {
    auto &graph = active_data();
    bool compute_original_step_info = has_flag(mask, step_info_mask::orig_step);
    bool compute_barrier_step_info = has_flag(mask, step_info_mask::barrier_step);
    if (compute_original_step_info)
        reset_part(kkt.step);
    if (compute_barrier_step_info)
        reset_part(kkt.barrier_step);

    graph.apply_forward(
        [&](data *d, data * /*next*/) {
            if (compute_original_step_info) {
                for (auto f : primal_fields) {
                    if (d->trial_prim_step[f].size() == 0) {
                        continue;
                    }
                    kkt.step.inf_prim_step = std::max(kkt.step.inf_prim_step, d->trial_prim_step[f].cwiseAbs().maxCoeff());
                }
                for (auto f : constr_fields) {
                    if (d->trial_dual_step[f].size() == 0) {
                        continue;
                    }
                    const scalar_t step = d->trial_dual_step[f].cwiseAbs().maxCoeff();
                    kkt.step.inf_dual_step = std::max(kkt.step.inf_dual_step, step);
                    if (in_field(f, ineq_constr_fields)) {
                        kkt.step.inf_ineq_dual_step = std::max(kkt.step.inf_ineq_dual_step, step);
                    } else {
                        kkt.step.inf_eq_dual_step = std::max(kkt.step.inf_eq_dual_step, step);
                    }
                }
            }
            if (compute_barrier_step_info) {
                for (auto f : primal_fields) {
                    if (d->trial_prim_step[f].size() == 0)
                        continue;
                    kkt.barrier_step.augmented_objective_fullstep_dec += d->dense().cost_jac_[f].dot(d->trial_prim_step[f]);
                }
                d->for_each<ineq_soft_constr_fields>([&](const soft_constr &c, soft_constr::data_map_t &sd) {
                    kkt.barrier_step.augmented_objective_fullstep_dec += c.objective_penalty_dir_deriv(sd);
                    kkt.barrier_step.search_barrier_dir_deriv += c.search_penalty_dir_deriv(sd);
                });
            }
        },
        true);
    if (compute_barrier_step_info) {
        kkt.barrier_step.ls_objective_fullstep_dec =
            kkt.barrier_step.augmented_objective_fullstep_dec - kkt.barrier_step.search_barrier_dir_deriv;
    }
}

void ns_sqp::update_primal_info(kkt_info &kkt, point_value_mask mask) {
    const bool include_primal = has_flag(mask, point_value_mask::primal);
    const bool include_barrier_objective = has_flag(mask, point_value_mask::barrier_objective);
    auto &graph = active_data();
    if (include_primal) {
        reset_part(kkt.primal);
    }
    if (include_barrier_objective) {
        reset_part(kkt.barrier_objective);
        if (!include_primal) {
        }
    }

    graph.apply_forward(
        [&](node_data *cur, node_data * /*next*/) {
            if (include_primal) {
                kkt.primal.inf_res = std::max(kkt.primal.inf_res, cur->inf_prim_res_);
                kkt.primal.res_l1 += cur->prim_res_l1_;
                kkt.primal.inf_comp = std::max(kkt.primal.inf_comp, cur->inf_comp_res_);
                cur->for_each<ineq_soft_constr_fields>([&](const soft_constr &c, soft_constr::data_map_t &sd) {
                    kkt.primal.inf_comp = std::max(kkt.primal.inf_comp, c.local_comp_residual_inf(sd));
                });
            }
            if (include_barrier_objective) {
                kkt.barrier_objective.cost += cur->cost();
                cur->for_each<ineq_soft_constr_fields>([&](const soft_constr &c, soft_constr::data_map_t &sd) {
                    kkt.barrier_objective.augmented_objective += c.objective_penalty(sd);
                    kkt.barrier_objective.barrier_value += c.search_penalty(sd);
                });
            }
        },
        true);
    if (include_barrier_objective) {
        kkt.barrier_objective.augmented_objective += kkt.barrier_objective.cost;
        kkt.barrier_objective.ls_objective = kkt.barrier_objective.augmented_objective -
                                             kkt.barrier_objective.barrier_value;
    }
}

void ns_sqp::update_stat_info(kkt_info &kkt) {
    auto &graph = active_data();
    reset_part(kkt.dual);
    const auto update_dual_inf_res = [&kkt](const row_vector &r) {
        kkt.dual.inf_res = std::max(kkt.dual.inf_res, max_abs_or_zero(r));
    };
    graph.apply_forward(
        [&](node_data *cur, node_data *next) {
            for (auto cf : constr_fields) {
                const auto &lam = cur->dense().dual_[cf];
                kkt.dual.n_constr += static_cast<size_t>(lam.size());
                if (lam.size() == 0) {
                    continue;
                }
                kkt.dual.lambda_l1 += lam.lpNorm<1>();
                const scalar_t lam_inf = lam.cwiseAbs().maxCoeff();
                kkt.dual.max_norm = std::max(kkt.dual.max_norm, lam_inf);
                if (in_field(cf, hard_constr_fields) || in_field(cf, soft_constr_fields)) {
                    kkt.dual.max_eq_norm = std::max(kkt.dual.max_eq_norm, lam_inf);
                } else if (in_field(cf, ineq_constr_fields)) {
                    kkt.dual.max_ineq_norm = std::max(kkt.dual.max_ineq_norm, lam_inf);
                }
            }
            cur->for_each<ineq_soft_constr_fields>([&](const soft_constr &c, soft_constr::data_map_t &sd) {
                kkt.dual.inf_res = std::max(kkt.dual.inf_res, c.local_stat_residual_inf(sd));
            });
            if (cur->dense().lag_jac_[__u].size() > 0) {
                update_dual_inf_res(cur->dense().lag_jac_[__u]);
            }
            if (next != nullptr) [[likely]] {
                row_vector tmp;
                tmp.conservativeResize(next->dense().lag_jac_[__x].cols());
                tmp.noalias() = next->dense().lag_jac_[__x] *
                                    utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                                cur->dense().lag_jac_[__y];
                if (tmp.size() > 0) {
                    update_dual_inf_res(tmp);
                }
            } else if (cur->dense().lag_jac_[__y].size() > 0) {
                update_dual_inf_res(cur->dense().lag_jac_[__y]);
            }
        },
        true);
    if (!in_restoration_phase() && kkt.dual.n_constr > 0) {
        const scalar_t s_d = std::max(static_cast<scalar_t>(settings.s_max),
                                      kkt.dual.lambda_l1 / static_cast<scalar_t>(kkt.dual.n_constr)) /
                             static_cast<scalar_t>(settings.s_max);
        kkt.dual.inf_res /= s_d;
    }
}

ns_sqp::result_type ns_sqp::update(size_t n_iter, bool verbose) {
    reset_profile();
    profiler_.start_update();
    settings.verbose = verbose;
    auto &graph = active_data();
    graph.no_except() = settings.no_except;
    settings.n_worker = graph.n_jobs();
    iter_last = {};
    kkt_last = initialize();
    try {
        filter_linesearch_data ls;
        ls.constr_vio_min = std::max(kkt_last.primal.res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);
        // ls.constr_vio_min = kkt_last.primal.res_l1 * settings.ls.constr_vio_min_frac;
        refresh_problem_flags();
        settings.max_iter = n_iter;
        for (size_t i_iter = iter_last.num_iter; i_iter < n_iter;) {
            fmt::println("======================== Iteration: {}", i_iter + 1);
            profiler_.start_iteration(i_iter);
            timed_block_start("sqp_single_iter");
            const scalar_t inf_prim_before = kkt_last.primal.inf_res;
            line_search_action action = sqp_iter(ls, kkt_last,
                                                 /*do_scaling=*/true, /*do_refinement=*/true);
            timed_block_end("sqp_single_iter");

            iter_last.num_iter = i_iter + 1;
            profiler_.finish_iteration(ls.step_cnt);

            if (verbose) {
                print_stats(kkt_last, iter_last, ls.step_cnt);
            }

            const bool tiny_step_trigger =
                action == line_search_action::failure &&
                ls.failure_reason == filter_linesearch_per_iter_data::failure_reason_t::tiny_step &&
                settings.restoration.enabled &&
                kkt_last.primal.inf_res > settings.prim_tol &&
                settings.ls.alpha_primal <= current_linesearch_alpha_min(ls);

            if (tiny_step_trigger) {
                auto resto = restoration_update(kkt_last, iter_last, ls);
                kkt_last = resto;
                iter_last = resto.iter;
                ls.reset_per_iter_data();
                if (iter_last.result == iter_result_t::restoration_failed ||
                    iter_last.result == iter_result_t::restoration_reached_max_iter ||
                    iter_last.result == iter_result_t::infeasible_stationary) {
                    break;
                }
                i_iter = iter_last.num_iter;
                continue;
            }

            if (kkt_last.dual.inf_res < settings.dual_tol &&
                kkt_last.primal.inf_res < settings.prim_tol &&
                kkt_last.primal.inf_comp < settings.comp_tol) {
                if (verbose)
                    fmt::print("Converged!\n");
                iter_last.result = iter_result_t::success;
                break;
            }

            if (settings.has_ipm_ineq && settings.ipm.mu_method == solver::ipm_config::monotonic_decrease) {
                bool mu_changed = false;
                while (kkt_last.primal.inf_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.dual.inf_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.primal.inf_comp < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold) {
                    settings.ipm.mu *= settings.ipm.mu_monotone_factor;
                    if (verbose)
                        fmt::print("Monotone decrease of mu: new mu = {:.3e}\n", settings.ipm.mu);
                    ls.points.clear();
                    mu_changed = true;
                }
                if (!mu_changed) {
                    bool prim_fail = kkt_last.primal.inf_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool dual_fail = kkt_last.dual.inf_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool comp_fail = kkt_last.primal.inf_comp >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    if (verbose)
                        fmt::print("Not using monotone decrease of mu: primal res {} threshold, dual res {} threshold, comp res {} threshold\n",
                                   prim_fail ? "exceeds" : "within",
                                   dual_fail ? "exceeds" : "within",
                                   comp_fail ? "exceeds" : "within",
                                   settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold);
                }
                if (mu_changed) {
                    // kkt_last carries accepted primal/dual residuals from the previous iterate,
                    // but its barrier-weighted objective summaries are now stale under the new mu.
                    update_primal_info(kkt_last, point_value_mask::barrier_objective);
                }
                settings.ipm.mu = std::max(settings.ipm.mu, 1e-11);
            }
            ++i_iter;
        }
        if (iter_last.result == iter_result_t::unknown) {
            iter_last.result = iter_result_t::exceed_max_iter;
        }
    } catch (...) {
        if (settings.no_except) {
            if (settings.verbose)
                fmt::print("Exception caught during SQP iterations. Terminating.\n");
        } else {
            throw;
        }
    }
    profiler_.finish_update(profile_report_);
    result_type result;
    static_cast<kkt_info &>(result) = kkt_last;
    result.iter = iter_last;
    return result;
}
} // namespace moto
