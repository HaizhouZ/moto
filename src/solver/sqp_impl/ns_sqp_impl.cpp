#include <moto/solver/ineq_soft.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/field_conversion.hpp>
#include <magic_enum/magic_enum.hpp>
#include <numeric>

// #define ENABLE_TIMED_BLOCK
#define SHOW_DETAIL_TIMING
#include <moto/utils/timed_block.hpp>

namespace moto {
namespace {

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

scalar_t max_abs_or_zero(const row_vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

void accumulate_constraint_objective_terms(ns_sqp::kkt_info &kkt,
                                           node_data &node,
                                           bool track_diag_scaling = false,
                                           bool include_local_residuals = true) {
    node.for_each<soft_constr_fields>([&](const soft_constr &c, soft_constr::data_map_t &sd) {
        kkt.objective += c.objective_penalty(sd);
        kkt.search_barrier_value += c.search_penalty(sd);
        kkt.obj_fullstep_dec += c.objective_penalty_dir_deriv(sd);
        kkt.search_barrier_dir_deriv += c.search_penalty_dir_deriv(sd);
        if (include_local_residuals) {
            kkt.inf_dual_res = std::max(kkt.inf_dual_res, c.local_stat_residual_inf(sd));
            kkt.inf_comp_res = std::max(kkt.inf_comp_res, c.local_comp_residual_inf(sd));
        }
    });
    node.for_each<ineq_constr_fields>([&](const ineq_constr &c, ineq_constr::data_map_t &id) {
        kkt.objective += c.objective_penalty(id);
        kkt.search_barrier_value += c.search_penalty(id);
        kkt.obj_fullstep_dec += c.objective_penalty_dir_deriv(id);
        kkt.search_barrier_dir_deriv += c.search_penalty_dir_deriv(id);
        if (include_local_residuals) {
            kkt.inf_dual_res = std::max(kkt.inf_dual_res, c.local_stat_residual_inf(id));
            kkt.inf_comp_res = std::max(kkt.inf_comp_res, c.local_comp_residual_inf(id));
        }
        if (track_diag_scaling) {
            auto *ipm = dynamic_cast<solver::ipm_constr::approx_data *>(&id);
            if (ipm != nullptr && ipm->diag_scaling.size() > 0) {
                kkt.max_diag_scaling = std::max(kkt.max_diag_scaling, ipm->diag_scaling.cwiseAbs().maxCoeff());
                kkt.log_slack_sum += ipm->slack_.array().log().sum();
                if (ipm->d_slack_.size() > 0) {
                    kkt.barrier_dir_deriv += (ipm->d_slack_.array() / ipm->slack_backup_.array()).sum();
                }
            }
        }
    });
}

void accumulate_dual_norms(ns_sqp::kkt_info &kkt,
                           const array_type<vector, constr_fields> &dual,
                           bool include_soft_eq_in_eq_norm) {
    for (auto cf : constr_fields) {
        const auto &lam = dual[cf];
        if (lam.size() == 0) {
            continue;
        }
        const scalar_t lam_inf = lam.cwiseAbs().maxCoeff();
        kkt.max_dual_norm = std::max(kkt.max_dual_norm, lam_inf);
        if (cf == __dyn || cf == __eq_x || cf == __eq_xu ||
            (include_soft_eq_in_eq_norm && (cf == __eq_x_soft || cf == __eq_xu_soft))) {
            kkt.max_eq_dual_norm = std::max(kkt.max_eq_dual_norm, lam_inf);
        } else if (cf == __ineq_x || cf == __ineq_xu) {
            kkt.max_ineq_dual_norm = std::max(kkt.max_ineq_dual_norm, lam_inf);
        }
    }
}

void accumulate_dual_norms_and_l1(ns_sqp::kkt_info &kkt,
                                  const array_type<vector, constr_fields> &dual,
                                  scalar_t &lambda_l1,
                                  size_t &n_constr) {
    for (auto cf : constr_fields) {
        const auto &lam = dual[cf];
        if (lam.size() == 0) {
            continue;
        }
        lambda_l1 += lam.lpNorm<1>();
        n_constr += static_cast<size_t>(lam.size());
        const scalar_t lam_inf = lam.cwiseAbs().maxCoeff();
        kkt.max_dual_norm = std::max(kkt.max_dual_norm, lam_inf);
        if (cf == __dyn || cf == __eq_x || cf == __eq_xu) {
            kkt.max_eq_dual_norm = std::max(kkt.max_eq_dual_norm, lam_inf);
        } else if (cf == __ineq_x || cf == __ineq_xu) {
            kkt.max_ineq_dual_norm = std::max(kkt.max_ineq_dual_norm, lam_inf);
        }
    }
}

void accumulate_primal_step_and_obj_dec(ns_sqp::kkt_info &kkt,
                                        ns_sqp::data &node,
                                        bool check_cost_size) {
    for (auto f : primal_fields) {
        if (node.trial_prim_step[f].size() == 0) {
            continue;
        }
        const scalar_t step_inf = node.trial_prim_step[f].cwiseAbs().maxCoeff();
        kkt.inf_prim_step = std::max(kkt.inf_prim_step, step_inf);
        if (check_cost_size && node.dense().cost_jac_[f].size() != node.trial_prim_step[f].size()) {
            throw std::runtime_error(fmt::format(
                "cost/trial step size mismatch on field {}: cost_jac size {}, trial_prim_step size {}, lag_jac size {}, problem uid {}",
                field::name(f),
                node.dense().cost_jac_[f].size(),
                node.trial_prim_step[f].size(),
                node.dense().lag_jac_[f].size(),
                node.problem().uid()));
        }
        kkt.obj_fullstep_dec += node.dense().cost_jac_[f].dot(node.trial_prim_step[f]);
    }
}

void accumulate_dual_step_info(ns_sqp::kkt_info &kkt,
                               const array_type<vector, constr_fields> &trial_dual_step) {
    for (auto f : constr_fields) {
        if (trial_dual_step[f].size() == 0) {
            continue;
        }
        const scalar_t step = trial_dual_step[f].cwiseAbs().maxCoeff();
        kkt.inf_dual_step = std::max(kkt.inf_dual_step, step);
        if (in_field(f, ineq_constr_fields)) {
            kkt.inf_ineq_dual_step = std::max(kkt.inf_ineq_dual_step, step);
        } else {
            kkt.inf_eq_dual_step = std::max(kkt.inf_eq_dual_step, step);
            if (f == __dyn) kkt.inf_dyn_dual_step = std::max(kkt.inf_dyn_dual_step, step);
            if (f == __eq_x) kkt.inf_eq_x_dual_step = std::max(kkt.inf_eq_x_dual_step, step);
            if (f == __eq_xu) kkt.inf_eq_xu_dual_step = std::max(kkt.inf_eq_xu_dual_step, step);
        }
    }
}

void accumulate_u_stationarity(const row_vector &r,
                               ns_sqp::kkt_info &kkt,
                               scalar_t &dual_res_l1,
                               size_t &n_dual_res) {
    kkt.inf_dual_res = std::max(kkt.inf_dual_res, max_abs_or_zero(r));
    dual_res_l1 += r.cwiseAbs().sum();
    n_dual_res += static_cast<size_t>(r.size());
}

void finalize_phase_objectives(ns_sqp::kkt_info &kkt) {
    kkt.penalized_obj = kkt.objective - kkt.search_barrier_value;
    kkt.penalized_obj_fullstep_dec = kkt.obj_fullstep_dec - kkt.search_barrier_dir_deriv;
}

template <class UFn, class XYFn>
void accumulate_w_stationarity(ns_sqp::storage_type &graph,
                               UFn &&on_u_residual,
                               XYFn &&on_xy_residual) {
    for (auto *n : graph.flatten_nodes()) {
        if (n->dense().lag_jac_[__u].size() > 0) {
            on_u_residual(n->dense().lag_jac_[__u]);
        }
    }

    graph.apply_forward(
        [&](node_data *cur, node_data *next) {
            if (next != nullptr) [[likely]] {
                row_vector tmp;
                tmp.conservativeResize(next->dense().lag_jac_[__x].cols());
                tmp.noalias() = next->dense().lag_jac_[__x] *
                                    utils::permutation_from_y_to_x(&cur->problem(), &next->problem()) +
                                cur->dense().lag_jac_[__y];
                if (tmp.size() > 0) {
                    on_xy_residual(tmp);
                }
            } else if (cur->dense().lag_jac_[__y].size() > 0) {
                on_xy_residual(cur->dense().lag_jac_[__y]);
            }
        },
        true);
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

void ns_sqp::update_phase_problem(data *d, node_data::update_mode mode) {
    if (using_restoration_overlay_graph() || !in_restoration_phase()) {
        d->update_approximation(mode, true);
        return;
    }
    d->update_approximation(mode, true);
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
            cur->update_approximation(node_data::update_mode::eval_val);
            // initialize the data
            if (!settings.ipm.warm_start || !settings.initialized)
                solver::ineq_soft::initialize(cur);
            settings.initialized = true;
        });
    }
    {
        auto phase_profile = profile_scope(profile_phase::initialize_derivative_eval);
        graph.for_each_parallel([](data *cur) {
            cur->update_approximation(node_data::update_mode::eval_derivatives);
        });
    }
    initialize_equality_multipliers();
    kkt_info kkt;
    {
        auto phase_profile = profile_scope(profile_phase::initialize_kkt);
        kkt = compute_kkt_info();
    }
    reset_scaling(); // clear scale vectors; will be recomputed on first iteration
    // print statistics header
    if (settings.verbose) {
        // print_scaling_info();
        print_stat_header();
        print_stats(kkt); // print initial stats
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
    });
    unscale_duals();
    graph.for_each_parallel([this](data *d) {
        d->backup_trial_state();
        if (use_normal_soft_phase() || in_restoration_phase()) {
            solver::ineq_soft::backup_trial_state(d);
        }
    });
    if (ctx.mu_changed) {
        ls.points.clear(); // the QP objective changed, so old filter points are no longer comparable
    }
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
            if (use_normal_soft_phase() || in_restoration_phase()) {
                solver::ineq_soft::restore_trial_state(d);
            }
            riccati_solver_->apply_affine_step(d, &settings);
            if (use_normal_soft_phase() || in_restoration_phase()) {
                solver::ineq_soft::apply_affine_step(d, &settings);
            }
            update_phase_problem(d, node_data::update_mode::eval_val);
        });
    detail_timed_block_end("apply_affine_step");
    }

    {
        auto res_profile = profile_scope(profile_phase::update_res_stat);
    detail_timed_block_start("update_res_stat");
    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking ||
        in_restoration_phase()) {
        graph.for_each_parallel([this](data *d) {
            update_phase_problem(d, in_restoration_phase() ? node_data::update_mode::eval_raw_derivatives
                                                           : node_data::update_mode::eval_derivatives);
        });
        ctx.trial = compute_kkt_info();
    } else {
        ctx.trial = compute_kkt_info(false);
    }
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
                update_phase_problem(d, in_restoration_phase() ? node_data::update_mode::eval_raw_derivatives
                                                               : node_data::update_mode::eval_derivatives);
            });
            ctx.trial = compute_kkt_info();
        }
        detail_timed_block_end("update_approx_accepted");
    }
    ctx.current = ctx.trial;
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
            if (use_normal_soft_phase() || in_restoration_phase()) {
                solver::ineq_soft::restore_trial_state(d);
            }
            update_phase_problem(d, in_restoration_phase() ? node_data::update_mode::eval_raw_derivatives
                                                           : node_data::update_mode::eval_derivatives);
        });
        ctx.current = compute_kkt_info();
        return line_search_action::failure;
    }

    // Line-search fallback strategies select a concrete backup step
    // (best trial or min step). Re-evaluate that selected step and accept
    // it so the control flow matches the line-search decision.
    [[maybe_unused]] bool has_fallback_derivatives = evaluate_trial_point(ls, ctx);
    accept_trial_point(ls, ctx);
    return line_search_action::accept;
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
                ctx.current = ctx.trial;
            return ctx.action;
        case line_search_action::retry_second_order_correction:
            second_order_correction();
            break;
        case line_search_action::backtrack:
            if (!ls.recompute_approx)
                return ctx.action;
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
        .current = kkt_current,
        .phase = in_restoration_phase() ? iteration_phase::restoration : iteration_phase::normal,
    };
    reset_ls_workers();
    solve_direction(ctx, do_scaling, gauss_newton);
    correct_direction(ctx, do_refinement);
    prepare_globalization(ls, ctx);
    line_search_action action = run_globalization(ls, ctx);
    kkt_current = ctx.current;
    return action;
}

ns_sqp::kkt_info ns_sqp::update(size_t n_iter, bool verbose) {
    reset_profile();
    profiler_.start_update();
    settings.verbose = verbose;
    auto &graph = active_data();
    graph.no_except() = settings.no_except;
    settings.n_worker = graph.n_jobs();
    kkt_last = initialize();
    try {
        filter_linesearch_data ls;
        ls.constr_vio_min = std::max(kkt_last.prim_res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);
        // ls.constr_vio_min = kkt_last.prim_res_l1 * settings.ls.constr_vio_min_frac;
        refresh_problem_flags();
        settings.max_iter = n_iter;
        for (size_t i_iter = kkt_last.num_iter; i_iter < n_iter;) {
            fmt::println("======================== Iteration: {}", i_iter + 1);
            profiler_.start_iteration(i_iter);
            timed_block_start("sqp_single_iter");
            const scalar_t inf_prim_before = kkt_last.inf_prim_res;
            line_search_action action = sqp_iter(ls, kkt_last,
                                                 /*do_scaling=*/true, /*do_refinement=*/true);
            timed_block_end("sqp_single_iter");

            kkt_last.num_iter = i_iter + 1;
            kkt_last.ls_steps = ls.step_cnt;
            profiler_.finish_iteration(ls.step_cnt);

            if (verbose) {
                print_stats(kkt_last);
            }

            const bool tiny_step_trigger =
                action == line_search_action::failure &&
                ls.failure_reason == filter_linesearch_per_iter_data::failure_reason_t::tiny_step &&
                settings.restoration.enabled &&
                kkt_last.inf_prim_res > settings.prim_tol &&
                settings.ls.alpha_primal <= current_linesearch_alpha_min(ls);

            if (tiny_step_trigger) {
                kkt_last = restoration_update(kkt_last, ls);
                ls.reset_per_iter_data();
                if (kkt_last.result == iter_result_t::restoration_failed ||
                    kkt_last.result == iter_result_t::restoration_reached_max_iter ||
                    kkt_last.result == iter_result_t::infeasible_stationary) {
                    break;
                }
                i_iter = kkt_last.num_iter;
                continue;
            }

            if (kkt_last.inf_dual_res < settings.dual_tol &&
                kkt_last.inf_prim_res < settings.prim_tol &&
                kkt_last.inf_comp_res < settings.comp_tol) {
                if (verbose)
                    fmt::print("Converged!\n");
                kkt_last.result = iter_result_t::success;
                break;
            }

            if (settings.has_ipm_ineq && settings.ipm.mu_method == solver::ipm_config::monotonic_decrease) {
                bool mu_changed = false;
                while (kkt_last.inf_prim_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_dual_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_comp_res < settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold) {
                    settings.ipm.mu *= settings.ipm.mu_monotone_factor;
                    if (verbose)
                        fmt::print("Monotone decrease of mu: new mu = {:.3e}\n", settings.ipm.mu);
                    ls.points.clear();
                    mu_changed = true;
                }
                if (!mu_changed) {
                    bool prim_fail = kkt_last.inf_prim_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool dual_fail = kkt_last.inf_dual_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    bool comp_fail = kkt_last.inf_comp_res >= settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    if (verbose)
                        fmt::print("Not using monotone decrease of mu: primal res {} threshold, dual res {} threshold, comp res {} threshold\n",
                                   prim_fail ? "exceeds" : "within",
                                   dual_fail ? "exceeds" : "within",
                                   comp_fail ? "exceeds" : "within",
                                   settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold);
                }
                settings.ipm.mu = std::max(settings.ipm.mu, 1e-11);
            }
            ++i_iter;
        }
        if (kkt_last.result == iter_result_t::unknown) {
            kkt_last.result = iter_result_t::exceed_max_iter;
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
    return kkt_last;
}

ns_sqp::kkt_info ns_sqp::update_minimal(size_t n_iter, bool verbose) {
    reset_profile();
    profiler_.start_update();
    settings.verbose = verbose;
    settings.in_restoration = false;
    clear_phase_graph_override();
    const bool old_rf_enabled = settings.rf.enabled;
    settings.rf.enabled = false;
    auto &graph = active_data();
    graph.no_except() = settings.no_except;
    settings.n_worker = graph.n_jobs();
    refresh_problem_flags();

    if (settings.verbose)
        fmt::print("Initialization for SQP...\n");
    if (!settings.ipm.warm_start)
        settings.ipm.mu = settings.ipm.mu0;

    const bool need_soft_init = !settings.ipm.warm_start || !settings.initialized;
    {
        auto profile = profile_scope(profile_phase::initialize_total);
        {
            auto phase_profile = profile_scope(profile_phase::initialize_setup_eval);
            graph.for_each_parallel([this, need_soft_init](data *cur) {
                cur->for_each_constr([this](const generic_func &c, func_approx_data &d) {
                    c.setup_workspace_data(d, &settings);
                });
                cur->update_approximation(node_data::update_mode::eval_val);
                if (need_soft_init)
                    solver::ineq_soft::initialize(cur);
            });
        }
        settings.initialized = true;
        {
            auto phase_profile = profile_scope(profile_phase::initialize_derivative_eval);
            graph.for_each_parallel([](data *cur) {
                cur->update_approximation(node_data::update_mode::eval_derivatives);
            });
        }
        {
            auto phase_profile = profile_scope(profile_phase::initialize_kkt);
            kkt_last = compute_kkt_info_minimal();
        }
    }
    reset_scaling();

    if (settings.verbose) {
        print_stat_header();
        print_stats(kkt_last);
    }

    try {
        filter_linesearch_data ls;
        ls.constr_vio_min =
            std::max(kkt_last.prim_res_l1 * settings.ls.constr_vio_min_frac, settings.prim_tol);
        settings.max_iter = n_iter;

        for (size_t i_iter = kkt_last.num_iter; i_iter < n_iter; ++i_iter) {
            auto iter_profile = profile_scope(profile_phase::sqp_iter_total);
            profiler_.start_iteration(i_iter + 1);
            timed_block_start("sqp_single_iter_minimal");

            iteration_context ctx{
                .current = kkt_last,
                .phase = iteration_phase::normal,
            };

            reset_ls_workers();
            solve_direction(ctx, /*do_scaling=*/false, /*gauss_newton=*/false);
            correct_direction(ctx, /*do_refinement=*/false);

            if (!settings.ls.enabled) {
                graph.for_each_parallel([this](data *d) {
                    riccati_solver_->finalize_dual_newton_step(d);
                });
                unscale_duals();
                detail_timed_block_start("apply_fullstep_minimal");
                graph.for_each_parallel([this](data *d) {
                    riccati_solver_->apply_affine_step(d, &settings);
                    solver::ineq_soft::apply_affine_step(d, &settings);
                    d->update_approximation(node_data::update_mode::eval_val);
                    d->update_approximation(node_data::update_mode::eval_derivatives);
                });
                detail_timed_block_end("apply_fullstep_minimal");
                ctx.current = compute_kkt_info_minimal();
                goto iter_done;
            }

            ls.reset_per_iter_data();
            ls.initial_alpha_primal = settings.ls.alpha_primal;
            ls.initial_alpha_dual = settings.ls.alpha_dual;
            ls.best_trial = filter_linesearch_data::trial();
            ls.merit_fullstep = std::numeric_limits<scalar_t>::infinity();
            ls.best_merit_trial = filter_linesearch_data::merit_trial{};

            graph.for_each_parallel([this](data *d) {
                riccati_solver_->finalize_dual_newton_step(d);
            });
            unscale_duals();
            graph.for_each_parallel([](data *d) {
                d->backup_trial_state();
                solver::ineq_soft::backup_trial_state(d);
            });
            if (ctx.mu_changed)
                ls.points.clear();

            while (true) {
                bool has_trial_derivatives = false;
                detail_timed_block_start("apply_affine_step_minimal");
                graph.for_each_parallel([this](data *d) {
                    d->restore_trial_state();
                    solver::ineq_soft::restore_trial_state(d);
                    riccati_solver_->apply_affine_step(d, &settings);
                    solver::ineq_soft::apply_affine_step(d, &settings);
                    d->update_approximation(node_data::update_mode::eval_val);
                });
                detail_timed_block_end("apply_affine_step_minimal");

                detail_timed_block_start("update_res_stat_minimal");
                if (settings.ls.method == linesearch_setting::search_method::merit_backtracking) {
                    detail_timed_block_start("trial_derivatives_minimal");
                    graph.for_each_parallel([](data *d) {
                        d->update_approximation(node_data::update_mode::eval_derivatives);
                    });
                    detail_timed_block_end("trial_derivatives_minimal");
                    detail_timed_block_start("trial_kkt_minimal");
                    ctx.trial = compute_kkt_info_minimal();
                    detail_timed_block_end("trial_kkt_minimal");
                    has_trial_derivatives = true;
                } else {
                    detail_timed_block_start("trial_kkt_minimal");
                    ctx.trial = compute_kkt_info_minimal(false);
                    detail_timed_block_end("trial_kkt_minimal");
                }
                detail_timed_block_end("update_res_stat_minimal");

                if (settings.ls.enabled && !ls.stop) {
                    if (settings.ls.method == linesearch_setting::search_method::merit_backtracking)
                        ctx.action = merit_linesearch(ls, ctx.trial, ctx.current);
                    else
                        ctx.action = filter_linesearch(ls, ctx.trial, ctx.current);
                } else {
                    ctx.action = line_search_action::accept;
                }

                switch (ctx.action) {
                case line_search_action::accept:
                    if (!has_trial_derivatives) {
                        detail_timed_block_start("update_approx_accepted_minimal");
                        if (settings.ls.method !=
                            linesearch_setting::search_method::merit_backtracking) {
                            detail_timed_block_start("accepted_derivatives_minimal");
                            graph.for_each_parallel([](data *d) {
                                d->update_approximation(node_data::update_mode::eval_derivatives);
                            });
                            detail_timed_block_end("accepted_derivatives_minimal");
                            detail_timed_block_start("accepted_kkt_minimal");
                            ctx.trial = compute_kkt_info_minimal();
                            detail_timed_block_end("accepted_kkt_minimal");
                        }
                        detail_timed_block_end("update_approx_accepted_minimal");
                        ctx.current = ctx.trial;
                    } else {
                        ctx.current = ctx.trial;
                    }
                    goto iter_done;
                case line_search_action::retry_second_order_correction:
                    second_order_correction();
                    break;
                case line_search_action::backtrack:
                    if (!ls.recompute_approx)
                        goto iter_done;
                    break;
                case line_search_action::failure:
                    ctx.action = handle_globalization_failure(ls, ctx);
                    if (ctx.action == line_search_action::accept ||
                        ctx.action == line_search_action::failure)
                        goto iter_done;
                    throw std::runtime_error("minimal line search failed");
                }
            }

        iter_done:
            timed_block_end("sqp_single_iter_minimal");
            profiler_.finish_iteration(ls.step_cnt);
            kkt_last = ctx.current;
            kkt_last.num_iter = i_iter + 1;
            kkt_last.ls_steps = ls.step_cnt;

            if (verbose)
                print_stats(kkt_last);

            if (kkt_last.inf_dual_res < settings.dual_tol &&
                kkt_last.inf_prim_res < settings.prim_tol &&
                kkt_last.inf_comp_res < settings.comp_tol) {
                if (verbose)
                    fmt::print("Converged!\n");
                kkt_last.result = iter_result_t::success;
                break;
            }

            if (settings.has_ineq_soft &&
                settings.ipm.mu_method == solver::ipm_config::monotonic_decrease) {
                bool mu_changed = false;
                while (kkt_last.inf_prim_res <
                           settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_dual_res <
                           settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold &&
                       kkt_last.inf_comp_res <
                           settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold) {
                    settings.ipm.mu *= settings.ipm.mu_monotone_factor;
                    if (verbose)
                        fmt::print("Monotone decrease of mu: new mu = {:.3e}\n",
                                   settings.ipm.mu);
                    ls.points.clear();
                    mu_changed = true;
                }
                if (!mu_changed && verbose) {
                    const bool prim_fail =
                        kkt_last.inf_prim_res >=
                        settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    const bool dual_fail =
                        kkt_last.inf_dual_res >=
                        settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    const bool comp_fail =
                        kkt_last.inf_comp_res >=
                        settings.ipm.mu * settings.ipm.mu_monotone_fraction_threshold;
                    fmt::print(
                        "Not using monotone decrease of mu: primal res {} threshold, dual res {} threshold, comp res {} threshold\n",
                        prim_fail ? "exceeds" : "within",
                        dual_fail ? "exceeds" : "within",
                        comp_fail ? "exceeds" : "within");
                }
                settings.ipm.mu = std::max(settings.ipm.mu, scalar_t(1e-11));
            }
        }
        if (kkt_last.result == iter_result_t::unknown)
            kkt_last.result = iter_result_t::exceed_max_iter;
    } catch (...) {
        settings.rf.enabled = old_rf_enabled;
        if (settings.no_except) {
            if (settings.verbose)
                fmt::print("Exception caught during minimal SQP iterations. Terminating.\n");
        } else {
            throw;
        }
    }
    profiler_.finish_update(profile_report_);
    settings.rf.enabled = old_rf_enabled;
    return kkt_last;
}

void ns_sqp::print_dual_res_breakdown() {
    // Decompose dual_res by field at the worst-case node.
    //
    // dual_res = max_k ||r_k||_inf where
    //   r_k = lag_jac_[__y]_k + lag_jac_[__x]_{k+1} * P   (cross-stage costate residual)
    //
    // We first find the stage k* that achieves dual_res, then decompose:
    //   r_k* = sum_cf r_{cf,k*}   where r_{cf,k} = J_{cf,y,k}^T λ_{cf,k} + J_{cf,x,k+1}^T λ_{cf,k+1}
    //
    // These per-field vectors sum to r_k* exactly, so their inf-norms are a true
    // decomposition of dual_res (signed cancellation may make sum < sum-of-norms).

    auto &graph = active_data();
    auto nodes = graph.flatten_nodes();

    // ── Pass 1: find node k* that achieves max cross-stage costate residual ──
    size_t worst_i = 0;
    scalar_t worst_y = 0.;
    scalar_t worst_u = 0.;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto *cur = nodes[i];
        node_data *next = (i + 1 < nodes.size()) ? nodes[i + 1] : nullptr;
        // u stationarity
        scalar_t u_res = cur->dense().lag_jac_[__u].cwiseAbs().maxCoeff();
        worst_u = std::max(worst_u, u_res);
        // cross-stage costate: mirrors compute_kkt_info
        if (next) {
            auto P = utils::permutation_from_y_to_x(&cur->problem(), &next->problem());
            row_vector total_r = next->dense().lag_jac_[__x] * P + cur->dense().lag_jac_[__y];
            scalar_t y_res = total_r.cwiseAbs().maxCoeff();
            if (y_res > worst_y) {
                worst_y = y_res;
                worst_i = i;
            }
        } else {
            scalar_t y_res = cur->dense().lag_jac_[__y].cwiseAbs().maxCoeff();
            if (y_res > worst_y) {
                worst_y = y_res;
                worst_i = i;
            }
        }
    }

    // ── Pass 2: decompose r_{k*} by field ────────────────────────────────────
    auto *cur = nodes[worst_i];
    node_data *next = (worst_i + 1 < nodes.size()) ? nodes[worst_i + 1] : nullptr;
    auto P = utils::permutation_from_y_to_x(&cur->problem(), next ? &next->problem() : &cur->problem());

    int ny = (int)cur->dense().lag_jac_[__y].cols();
    int nx_next = next ? (int)next->dense().lag_jac_[__x].cols() : ny;

    // per-field cross-stage vector at k* (signed, not yet normed)
    std::map<field_t, scalar_t> cross_norm; // ||r_{cf,k*}||_inf
    std::map<field_t, scalar_t> u_norm;     // ||J_{cf,u,k*}^T λ_{cf,k*}||_inf

    for (field_t cf : constr_fields) {
        auto &approx = cur->dense().approx_[cf];
        if (approx.v_.size() == 0)
            continue;
        const auto &lam = cur->dense().dual_[cf];
        if (lam.size() == 0)
            continue;

        // u-contribution at k*
        if (!approx.jac_[__u].is_empty()) {
            row_vector tmp(approx.jac_[__u].cols());
            tmp.setZero();
            approx.jac_[__u].right_T_times(lam, tmp);
            u_norm[cf] = tmp.cwiseAbs().maxCoeff();
        }

        // cross-stage y-contribution: J_{cf,y,k*}^T λ_{cf,k*} + J_{cf,x,k*+1}^T λ_{cf,k*+1}
        row_vector r = row_vector::Zero(nx_next);
        if (!approx.jac_[__y].is_empty()) {
            row_vector tmp(ny);
            tmp.setZero();
            approx.jac_[__y].right_T_times(lam, tmp);
            r += (tmp * P).eval();
        }
        if (next) {
            auto &next_approx = next->dense().approx_[cf];
            const auto &lam_next = next->dense().dual_[cf];
            if (!next_approx.jac_[__x].is_empty() && lam_next.size() > 0) {
                row_vector tmp(nx_next);
                tmp.setZero();
                next_approx.jac_[__x].right_T_times(lam_next, tmp);
                r += tmp;
            }
        }
        cross_norm[cf] = r.cwiseAbs().maxCoeff();
    }

    // cost-only: lag_jac_[__y]_k* + lag_jac_[__x]_{k*+1} * P  minus field contributions
    // (we just report total which mirrors dual_res exactly)
    fmt::print("  dual_res breakdown at worst node {} (y={:.3e} u={:.3e}):\n",
               worst_i, worst_y, worst_u);
    fmt::print("    [{:12s}]: cross-stage-y={:.3e}  u={:.3e}\n", "total", worst_y, worst_u);
    for (field_t cf : constr_fields) {
        scalar_t cu = u_norm.count(cf) ? u_norm[cf] : 0.;
        scalar_t cy = cross_norm.count(cf) ? cross_norm[cf] : 0.;
        if (cu > 1e-10 || cy > 1e-10)
            fmt::print("    [{:12s}]: cross-stage-y={:.3e}  u={:.3e}\n", field::name(cf), cy, cu);
    }
}

ns_sqp::kkt_info ns_sqp::compute_kkt_info_minimal(bool update_dual_res) {
    kkt_info kkt;
    scalar_t lambda_l1 = 0.;
    size_t n_constr = 0;
    scalar_t dual_res_l1 = 0.;
    size_t n_dual_res = 0;
    auto &graph = active_data();

    for (auto *n : graph.flatten_nodes()) {
        kkt.cost += n->cost();
        kkt.inf_prim_res = std::max(kkt.inf_prim_res, n->inf_prim_res_);
        kkt.prim_res_l1 += n->prim_res_l1_;
        if (update_dual_res && n->dense().lag_jac_[__u].size() > 0) {
            const auto &r_u = n->dense().lag_jac_[__u];
            kkt.inf_dual_res = std::max(kkt.inf_dual_res, max_abs_or_zero(r_u));
            dual_res_l1 += r_u.cwiseAbs().sum();
            n_dual_res += static_cast<size_t>(r_u.size());
        }
        kkt.inf_comp_res = std::max(kkt.inf_comp_res, n->inf_comp_res_);
        if (update_dual_res) {
            accumulate_dual_norms_and_l1(kkt, n->dense().dual_, lambda_l1, n_constr);
        }
        for (auto f : primal_fields) {
            if (n->trial_prim_step[f].size() == 0) {
                continue;
            }
            kkt.inf_prim_step =
                std::max(kkt.inf_prim_step, max_abs_or_zero(n->trial_prim_step[f]));
            kkt.obj_fullstep_dec += n->dense().cost_jac_[f].dot(n->trial_prim_step[f]);
        }
        n->for_each<ineq_soft_constr_fields>([&](const soft_constr &, soft_constr::data_map_t &sd) {
            auto *id = dynamic_cast<solver::ipm_constr::approx_data *>(&sd);
            if (id == nullptr || id->ipm_cfg == nullptr)
                return;
            kkt.log_slack_sum += id->slack_.array().log().sum();
            if (id->d_slack_.size() > 0)
                kkt.barrier_dir_deriv += (id->d_slack_.array() / id->slack_backup_.array()).sum();
            if (id->diag_scaling.size() > 0)
                kkt.max_diag_scaling =
                    std::max(kkt.max_diag_scaling, id->diag_scaling.cwiseAbs().maxCoeff());
        });
        accumulate_dual_step_info(kkt, n->trial_dual_step);
    }

    if (update_dual_res) {
        graph.apply_forward(
            [&](node_data *cur, node_data *next) {
                if (next != nullptr) [[likely]] {
                    row_vector tmp;
                    tmp.conservativeResize(next->dense().lag_jac_[__x].cols());
                    tmp.noalias() = next->dense().lag_jac_[__x] *
                                        utils::permutation_from_y_to_x(&cur->problem(),
                                                                       &next->problem()) +
                                    cur->dense().lag_jac_[__y];
                    if (tmp.size() > 0) {
                        kkt.inf_dual_res = std::max(kkt.inf_dual_res, max_abs_or_zero(tmp));
                        dual_res_l1 += tmp.cwiseAbs().sum();
                        n_dual_res += static_cast<size_t>(tmp.size());
                    }
                } else if (cur->dense().lag_jac_[__y].size() > 0) {
                    const auto &r_y = cur->dense().lag_jac_[__y];
                    kkt.inf_dual_res = std::max(kkt.inf_dual_res, max_abs_or_zero(r_y));
                    dual_res_l1 += r_y.cwiseAbs().sum();
                    n_dual_res += static_cast<size_t>(r_y.size());
                }
            },
            true);
        if (n_constr > 0) {
            const scalar_t s_d =
                std::max(settings.s_max,
                         lambda_l1 / static_cast<scalar_t>(n_constr)) /
                settings.s_max;
            kkt.inf_dual_res /= s_d;
        }
        if (n_dual_res > 0) {
            kkt.avg_dual_res = dual_res_l1 / static_cast<scalar_t>(n_dual_res);
        }
    }

    kkt.objective = kkt.cost - settings.ipm.mu * kkt.log_slack_sum;
    kkt.penalized_obj = kkt.objective;
    kkt.penalized_obj_fullstep_dec = kkt.obj_fullstep_dec - settings.ipm.mu * kkt.barrier_dir_deriv;
    return kkt;
}

ns_sqp::kkt_info ns_sqp::compute_kkt_info(bool update_dual_res) {
    if (in_restoration_phase()) {
        if (using_restoration_overlay_graph()) {
            kkt_info kkt;
            scalar_t dual_res_l1 = 0.;
            size_t n_dual_res = 0;
            auto &graph = active_data();

            for (auto n : graph.flatten_nodes()) {
                kkt.cost += n->cost();
                kkt.objective += n->cost();
                kkt.inf_prim_res = std::max(kkt.inf_prim_res, n->inf_prim_res_);
                kkt.prim_res_l1 += n->prim_res_l1_;

                if (update_dual_res && n->dense().lag_jac_[__u].size() > 0) {
                    accumulate_u_stationarity(n->dense().lag_jac_[__u], kkt, dual_res_l1, n_dual_res);
                }
                // In restoration, the elastic overlay wrappers own the relevant
                // local stationarity and complementarity residuals. Include them
                // here so the reported phase KKT metrics reflect the active
                // restoration problem instead of printing zeros for r(dual)/r(comp).
                accumulate_constraint_objective_terms(kkt, *n, false, true);
                accumulate_dual_norms(kkt, n->dense().dual_, true);
                accumulate_primal_step_and_obj_dec(kkt, *n, false);
                accumulate_dual_step_info(kkt, n->trial_dual_step);
            }

            if (update_dual_res) {
                accumulate_w_stationarity(
                    graph,
                    [&](const row_vector &r) {
                        accumulate_u_stationarity(r, kkt, dual_res_l1, n_dual_res);
                    },
                    [&](const row_vector &r) {
                        accumulate_u_stationarity(r, kkt, dual_res_l1, n_dual_res);
                    });
                if (n_dual_res > 0) {
                    kkt.avg_dual_res = dual_res_l1 / static_cast<scalar_t>(n_dual_res);
                }
            }
            finalize_phase_objectives(kkt);
            return kkt;
        }

        throw std::runtime_error("restoration phase without overlay graph is no longer supported");
    }

    kkt_info kkt;
    field_t max_field;
    vector max_step;
    scalar_t lambda_l1 = 0.;
    size_t n_constr = 0;
    scalar_t dual_res_l1 = 0.;
    size_t n_dual_res = 0;
    auto &graph = active_data();
    for (auto n : graph.flatten_nodes()) {
        kkt.cost += n->cost();
        kkt.inf_prim_res = std::max(kkt.inf_prim_res, n->inf_prim_res_);
        kkt.prim_res_l1 += n->prim_res_l1_;
        if (update_dual_res && n->dense().lag_jac_[__u].size() > 0) {
            accumulate_u_stationarity(n->dense().lag_jac_[__u], kkt, dual_res_l1, n_dual_res);
        }
        kkt.inf_comp_res = std::max(kkt.inf_comp_res, n->inf_comp_res_);
        if (update_dual_res) {
            accumulate_dual_norms_and_l1(kkt, n->dense().dual_, lambda_l1, n_constr);
        }
        accumulate_constraint_objective_terms(kkt, *n, true);
        accumulate_primal_step_and_obj_dec(kkt, *n, true);
        accumulate_dual_step_info(kkt, n->trial_dual_step);
    }
    if (update_dual_res) {
        accumulate_w_stationarity(
            graph,
            [&](const row_vector &r) { accumulate_u_stationarity(r, kkt, dual_res_l1, n_dual_res); },
            [&](const row_vector &r) { accumulate_u_stationarity(r, kkt, dual_res_l1, n_dual_res); });
        // IPOPT-style dual scaling: s_d = max(s_max, ||λ||_1 / n_constr) / s_max
        if (n_constr > 0) {
            scalar_t s_d = std::max(settings.s_max, lambda_l1 / static_cast<scalar_t>(n_constr)) / settings.s_max;
            kkt.inf_dual_res /= s_d;
        }
        if (n_dual_res > 0)
            kkt.avg_dual_res = dual_res_l1 / static_cast<scalar_t>(n_dual_res);
    }
    kkt.objective = kkt.cost;
    kkt.search_barrier_value = settings.ipm.mu * kkt.log_slack_sum;
    kkt.search_barrier_dir_deriv = settings.ipm.mu * kkt.barrier_dir_deriv;
    finalize_phase_objectives(kkt);
    return kkt;
}

ns_sqp::kkt_info ns_sqp::compute_kkt_info_for_phase(iteration_phase phase, bool update_dual_res) {
    const bool old_resto = settings.in_restoration;
    settings.in_restoration = phase == iteration_phase::restoration;
    kkt_info kkt = compute_kkt_info(update_dual_res);
    settings.in_restoration = old_resto;
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

    auto &graph = active_data();
    graph.apply_forward([&](node_data *cur_nd, node_data *next_nd) {
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
                            // J.row(i) * F_u  and  J.row(i) * F_x  via sparse_mat API
                            // (use row_vector temporaries — only (row_vector, row_vector) is
                            //  in the explicit instantiation list for right_times)
                            row_vector Jrow = J.row(i);
                            row_vector tmp_u = row_vector::Zero(nu);
                            row_vector tmp_x = row_vector::Zero(nx);
                            d->F_u.right_times<false>(Jrow, tmp_u);
                            d->F_x.right_times<false>(Jrow, tmp_x);
                            ineq_jac_u.row(n_active) += tmp_u;
                            ineq_jac_x.row(n_active) += tmp_x;
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
            if (nz_x > 0) {
                matrix F_x_Z_x(ny, nz_x);
                d->F_x.times<false>(Z_x, F_x_Z_x);
                FZ.leftCols(nz_x).noalias() = F_x_Z_x;
            }
            d->F_u.dump_into(FZ.rightCols(nu));
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
    auto &graph = active_data();
    for (auto n : graph.flatten_nodes()) {
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
            const auto &jac = n->dense().lag_jac_[pf];
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
