#include <moto/solver/ns_sqp.hpp>
#include <type_cast.hpp>

#include <nanobind/stl/bind_vector.h>

#include <enum_export.hpp>
using namespace moto;

NB_MAKE_OPAQUE(std::vector<stage_ocp_ptr_t>);

void register_submodule_ns_sqp(nb::module_ &m) {

    nb::class_<ns_sqp> sqp(
        m, "sqp",
        "Stage-structured nonlinear OCP model and nonsmooth SQP solver.");
    m.attr("ns_sqp_impl") = sqp;
    nb::bind_vector<std::vector<stage_ocp_ptr_t>>(
        sqp, "stage_list", "Mutable ordered collection of authored OCP stages.");
    sqp.def(nb::init<size_t>(), nb::arg("n_job") = 4,
            "Constructor for the SQP solver with a specified number of jobs")
        .def_prop_ro("st", [](ns_sqp &self) { return self.st(); }, "Initial graph boundary")
        .def_prop_ro("ed", [](ns_sqp &self) { return self.ed(); }, "Graph terminal boundary")
        .def_prop_ro(
            "stages",
            [](ns_sqp &self) -> auto & { return self.stages(); },
            nb::rv_policy::reference_internal,
            nb::sig("def stages(self) -> stage_list"),
            "Mutable ordered graph-owned stage vector")
        .def("update", [](ns_sqp &self, size_t n_iter, bool verbose, bool profile) {
            nb::gil_scoped_release rel;
            return self.update(n_iter, verbose, profile);
        }, nb::arg("n_iter") = 1, nb::arg("verbose") = true, nb::arg("profile") = false,
           "Update the SQP solver for a given number of iterations")
        .def("get_profile_report", &ns_sqp::profile, "Get the latest SQP wall-clock profile report")
        .def_prop_ro("n_job", &ns_sqp::n_jobs,
                     "Effective maximum number of SQP worker threads")
        .def_ro("settings", &ns_sqp::settings,
                nb::for_getter(nb::sig("def settings(self) -> settings_type")),
                "Top-level SQP solver settings")
        .def_ro("linear_solve_last", &ns_sqp::linear_solve_last,
                nb::for_getter(nb::sig("def linear_solve_last(self) -> linear_solve_info")),
                "Diagnostics for the most recent Newton direction");

    nb::class_<ns_sqp::ipm_config>(
        sqp, "ipm_config",
        "Interior-point method options for inequality constraints.")
        .def_rw("mu0", &ns_sqp::ipm_config::mu0, "Initial barrier parameter for the IPM solver")
        .def_rw("warm_start", &ns_sqp::ipm_config::warm_start, "Whether to warm start the IPM solver")
        .def_rw("mu_method", &ns_sqp::ipm_config::mu_method,
                nb::for_getter(nb::sig("def mu_method(self) -> adaptive_mu_t")),
                "Adaptive mu method for the IPM solver")
        .def_rw("mu_monotone_fraction_threshold", &ns_sqp::ipm_config::mu_monotone_fraction_threshold, "Threshold for monotone decrease of mu (smaller is more likely to use monotone decrease)")
        .def_rw("mu_monotone_factor", &ns_sqp::ipm_config::mu_monotone_factor, "Factor for monotone decrease of mu (smaller -> faster decrease)")
        .def_rw("globalization", &ns_sqp::ipm_config::globalization, "Whether to use globalization in the IPM solver");

    nb::class_<ns_sqp::regularization_settings>(
        sqp, "regularization_settings",
        "Adaptive primal regularization and Newton-direction validation options.")
        .def_rw("enabled", &ns_sqp::regularization_settings::enabled,
                "Whether failed or inaccurate Newton directions are retried with primal regularization (default: true)")
        .def_rw("validate_direction", &ns_sqp::regularization_settings::validate_direction,
                "Whether to form normalized full recovered KKT residuals and reject inaccurate directions (default: false)")
        .def_rw("initial", &ns_sqp::regularization_settings::initial,
                "Initial positive primal regularization after an unregularized attempt fails (default: 1e-4)")
        .def_rw("increase_factor", &ns_sqp::regularization_settings::increase_factor,
                "Multiplier applied to primal regularization between retry attempts (default: 10)")
        .def_rw("decrease_factor", &ns_sqp::regularization_settings::decrease_factor,
                "Multiplier applied to the last successful regularization when seeding a later solve (default: 1/3)")
        .def_rw("maximum", &ns_sqp::regularization_settings::maximum,
                "Upper bound for an attempted primal regularization (default: 1e8)")
        .def_rw("max_attempts", &ns_sqp::regularization_settings::max_attempts,
                "Maximum Newton solve attempts per direction, including the unregularized attempt (default: 14)")
        .def_rw("residual_tolerance", &ns_sqp::regularization_settings::residual_tolerance,
                "Acceptance tolerance for normalized full direction residuals when validation is enabled (default: 1e-6)");
    nb::enum_<ns_sqp::linear_solve_status>(
        sqp, "linear_solve_status",
        "Acceptance status of a Newton-direction linear solve.")
        .value("success", ns_sqp::linear_solve_status::success)
        .value("factorization_failed", ns_sqp::linear_solve_status::factorization_failed)
        .value("inaccurate_direction", ns_sqp::linear_solve_status::inaccurate_direction)
        .value("inconsistent_equalities", ns_sqp::linear_solve_status::inconsistent_equalities)
        .value("nonfinite_direction", ns_sqp::linear_solve_status::nonfinite_direction);
    nb::class_<ns_sqp::linear_solve_info>(
        sqp, "linear_solve_info",
        "Diagnostics from the most recent Newton-direction linear solve.")
        .def_ro("status", &ns_sqp::linear_solve_info::status,
                nb::for_getter(nb::sig("def status(self) -> linear_solve_status")),
                "Acceptance status of the most recent Newton direction")
        .def_ro("attempts", &ns_sqp::linear_solve_info::attempts,
                "Number of Newton solve attempts used by the most recent direction")
        .def_ro("regularization", &ns_sqp::linear_solve_info::regularization,
                "Primal regularization used by the most recent solve attempt")
        .def_ro("stationarity_residual", &ns_sqp::linear_solve_info::stationarity_residual,
                "Maximum normalized recovered-stationarity residual from final direction validation")
        .def_ro("equality_residual", &ns_sqp::linear_solve_info::equality_residual,
                "Maximum normalized hard-equality residual from final direction validation")
        .def_ro("inequality_residual", &ns_sqp::linear_solve_info::inequality_residual,
                "Maximum normalized inequality, soft-equality, or restoration residual from final direction validation");

    nb::class_<ns_sqp::iterative_refinement_setting> rf_setting(
        sqp, "iterative_refinement_setting",
        "Residual tolerances and iteration limits for iterative refinement.");
    rf_setting.def_rw("enabled", &ns_sqp::iterative_refinement_setting::enabled, "Whether to use iterative refinement")
        .def_rw("max_iters", &ns_sqp::iterative_refinement_setting::max_iters, "Maximum number of iterative refinement iterations")
        .def_rw("prim_res_tol", &ns_sqp::iterative_refinement_setting::prim_res_tol, "Primal residual tolerance for iterative refinement")
        .def_rw("dual_res_tol", &ns_sqp::iterative_refinement_setting::dual_res_tol, "Dual residual tolerance for iterative refinement");

    nb::class_<ns_sqp::restoration_settings> restoration_setting(
        sqp, "restoration_settings",
        "Feasibility-restoration phase configuration.");
    restoration_setting
        .def_rw("enabled", &ns_sqp::restoration_settings::enabled, "Whether restoration is enabled")
        .def_rw("max_iter", &ns_sqp::restoration_settings::max_iter, "Maximum number of restoration iterations")
        .def_rw("rho_u", &ns_sqp::restoration_settings::rho_u, "Restoration proximal weight on u")
        .def_rw("rho_y", &ns_sqp::restoration_settings::rho_y, "Restoration proximal weight on y")
        .def_rw("rho_eq", &ns_sqp::restoration_settings::rho_eq, "Elastic penalty weight for restoration equalities")
        .def_rw("rho_ineq", &ns_sqp::restoration_settings::rho_ineq, "Elastic penalty weight for restoration inequalities")
        .def_rw("restoration_improvement_frac", &ns_sqp::restoration_settings::restoration_improvement_frac, "Required fraction of primal infeasibility improvement to accept restoration exit")
        .def_rw("alpha_min_factor", &ns_sqp::restoration_settings::alpha_min_factor, "Tiny-step trigger factor used before entering restoration")
        .def_rw("bound_mult_reset_threshold", &ns_sqp::restoration_settings::bound_mult_reset_threshold, "Reset copied-back bound multipliers when they exceed this threshold")
        .def_rw("constr_mult_reset_threshold", &ns_sqp::restoration_settings::constr_mult_reset_threshold, "Reset copied-back equality multipliers when they exceed this threshold");

    nb::class_<ns_sqp::equality_multiplier_init_settings> eq_init_setting(
        sqp, "equality_multiplier_init_settings",
        "Equality-multiplier recovery and initialization options.");
    eq_init_setting
        .def_rw("enabled", &ns_sqp::equality_multiplier_init_settings::enabled,
                "Master switch for equality and soft-equality multiplier recovery (default: true)")
        .def_rw("recover_on_warm_start", &ns_sqp::equality_multiplier_init_settings::recover_on_warm_start,
                "Whether warm SQP initialization recovers equality multipliers instead of preserving them (default: true)")
        .def_rw("rebuild_after_restoration_exit", &ns_sqp::equality_multiplier_init_settings::rebuild_after_restoration_exit,
                "Whether to recover equality multipliers after restoration exits successfully (default: true)")
        .def_rw("rho_eq", &ns_sqp::equality_multiplier_init_settings::rho_eq,
                "PMM penalty used for equality-type constraints in the equality-init overlay (default: 1)")
        .def_rw("rf", &ns_sqp::equality_multiplier_init_settings::rf,
                nb::for_getter(nb::sig("def rf(self) -> iterative_refinement_setting")),
                "Dedicated iterative-refinement settings used only during equality-multiplier recovery (disabled by default)");

    auto ls_config_base = nb::class_<solver::linesearch_config>(m, "linesearch_config");
    ls_config_base.def_rw("update_alpha_dual", &solver::linesearch_config::update_alpha_dual, "Whether to update the dual step size during line search")
        .def_rw("eq_dual_alpha_source", &solver::linesearch_config::eq_dual_alpha_source, "Source for dual step size for equality constraints")
        .def_rw("ineq_dual_alpha_source", &solver::linesearch_config::ineq_dual_alpha_source, "Source for dual step size for inequality constraints");

    moto::export_enum<solver::linesearch_config::dual_alpha_source>(ls_config_base);

    nb::class_<ns_sqp::linesearch_setting, solver::linesearch_config> ls_setting(
        sqp, "linesearch_setting",
        "Filter or merit-backtracking globalization configuration.");
    ls_setting.def_rw("enabled", &ns_sqp::linesearch_setting::enabled, "Whether to use line search")
        .def_rw("max_steps", &ns_sqp::linesearch_setting::max_steps, "Optional maximum number of backtracking reductions; zero uses the computed minimum step only")
        .def_rw("failure_strategy", &ns_sqp::linesearch_setting::failure_strategy,
                nb::for_getter(nb::sig("def failure_strategy(self) -> failure_backup_strategy")),
                "Line search failure backup strategy")
        .def_rw("on_failure", &ns_sqp::linesearch_setting::on_failure,
                nb::for_getter(nb::sig("def on_failure(self) -> on_failure_action")),
                "Action to take after line search exhausts max_steps")
        .def_rw("method", &ns_sqp::linesearch_setting::method,
                nb::for_getter(nb::sig("def method(self) -> search_method")),
                "Line search method: filter (default) or merit_backtracking")
        .def_rw("primal_gamma", &ns_sqp::linesearch_setting::primal_gamma, "Primal improvement requirement for the filter (higher is stricter)")
        .def_rw("dual_gamma", &ns_sqp::linesearch_setting::dual_gamma, "Objective improvement requirement for the filter (higher is stricter)")
        .def_rw("constr_vio_min_frac", &ns_sqp::linesearch_setting::constr_vio_min_frac, "Threshold for switching condition (fraction of initial primal residual)")
        .def_rw("armijo_dec_frac", &ns_sqp::linesearch_setting::armijo_dec_frac, "Sufficient decrease tolerance (eta in Armijo condition), smaller -> more strict decrease requirement")
        .def_rw("s_phi", &ns_sqp::linesearch_setting::s_phi, "IPOPT switching condition exponent on objective decrease (s_phi in IPOPT paper, Section 3.3)")
        .def_rw("s_theta", &ns_sqp::linesearch_setting::s_theta, "IPOPT switching condition exponent on constraint violation (s_theta in IPOPT paper, Section 3.3)")
        .def_rw("alpha_min_frac", &ns_sqp::linesearch_setting::alpha_min_frac, "IPOPT gamma_alpha safety factor for the computed minimum filter step (default: 0.05)")
        .def_rw("watchdog_shortened_iter_trigger", &ns_sqp::linesearch_setting::watchdog_shortened_iter_trigger, "Consecutive accepted shortened steps before starting the IPOPT watchdog; zero disables it (default: 10)")
        .def_rw("watchdog_trial_iter_max", &ns_sqp::linesearch_setting::watchdog_trial_iter_max, "Maximum provisional watchdog iterations before restoring its reference iterate (default: 3)")
        .def_rw("merit_sigma", &ns_sqp::linesearch_setting::merit_sigma, "Merit backtracking: weight on ||dual residual||^2 relative to ||constraint violation||^2 (default 1.0)")
        .def_rw("enable_flat_obj_accept", &ns_sqp::linesearch_setting::enable_flat_obj_accept, "Accept step when objective is flat, iterate is nearly feasible, and step is non-trivial")
        .def_rw("flat_obj_dec_tol", &ns_sqp::linesearch_setting::flat_obj_dec_tol, "Absolute full-step decrease below which the objective is considered flat")
        .def_rw("flat_obj_prim_tol", &ns_sqp::linesearch_setting::flat_obj_prim_tol, "Primal residual must be below this for flat-objective accept")
        .def_rw("flat_obj_step_tol", &ns_sqp::linesearch_setting::flat_obj_step_tol, "Step norm must exceed this for flat-objective accept (ensures non-trivial step)");

    ls_setting.def_rw("backtrack_scheme", &ns_sqp::linesearch_setting::backtrack_scheme,
        nb::for_getter(nb::sig("def backtrack_scheme(self) -> backtrack_scheme")),
        "Backtracking scheme: geometric (default) or linspace")
        .def_rw("backtrack_factor", &ns_sqp::linesearch_setting::backtrack_factor, "Geometric reduction factor applied to alpha at each backtracking step");

    moto::export_enum<ns_sqp::linesearch_setting::failure_backup_strategy>(
        ls_setting, "Fallback trial point selected after line-search failure.");
    moto::export_enum<ns_sqp::linesearch_setting::on_failure_action>(
        ls_setting, "Action taken when line search cannot accept a trial step.");
    moto::export_enum<ns_sqp::linesearch_setting::backtrack_scheme_t>(
        sqp, "Step-size sequence used during line-search backtracking.");
    moto::export_enum<ns_sqp::linesearch_setting::search_method>(
        sqp, "Globalization method used to accept or reject SQP steps.");
    nb::enum_<ns_sqp::initial_state_mode>(
        sqp, "initial_state_mode",
        "Whether the initial state is fixed or optimized.")
        .value("fixed", ns_sqp::initial_state_mode::fixed)
        .value("optimized", ns_sqp::initial_state_mode::optimized);
    nb::class_<ns_sqp::settings_t>(
        sqp, "settings_type",
        "Top-level SQP configuration, available from ``sqp.settings``.")
        .def_ro("mu", &ns_sqp::settings_t::mu, "Barrier parameter for the IPM solver")
        .def_rw("ipm_conditional_corrector", &ns_sqp::settings_t::ipm_conditional_corrector, "Whether to use conditional corrector in the IPM solver")
        .def_prop_ro("ipm", [](ns_sqp::settings_t &self) -> auto & { return self.ipm; },
                     nb::sig("def ipm(self) -> ipm_config"), "IPM settings")
        .def_rw("rf", &ns_sqp::settings_t::rf,
                nb::for_getter(nb::sig("def rf(self) -> iterative_refinement_setting")),
                "Iterative refinement settings")
        .def_rw("regularization", &ns_sqp::settings_t::regularization,
                nb::for_getter(nb::sig("def regularization(self) -> regularization_settings")),
                "Adaptive Newton direction safeguards")
        .def_prop_ro("restoration", [](ns_sqp::settings_t &self) -> auto & { return self.restoration; },
                     nb::sig("def restoration(self) -> restoration_settings"), "Restoration settings")
        .def_prop_ro("eq_init", [](ns_sqp::settings_t &self) -> auto & { return self.eq_init; },
                     nb::sig("def eq_init(self) -> equality_multiplier_init_settings"),
                     "Equality multiplier initialization settings")
        .def_rw("initial_state", &ns_sqp::settings_t::initial_state,
                nb::for_getter(nb::sig("def initial_state(self) -> initial_state_mode")),
                "Initial-state treatment: fixed (default) or optimized through an internal virtual stage")
        .def_prop_ro("ls", [](ns_sqp::settings_t &self) -> auto & { return self.ls; },
                     nb::sig("def ls(self) -> linesearch_setting"), "Line search settings")
        .def_rw("scaling", &ns_sqp::settings_t::scaling,
                nb::for_getter(nb::sig("def scaling(self) -> scaling_settings")),
                "Jacobian scaling settings")
        .def_rw("no_except", &ns_sqp::settings_t::no_except, "Whether to suppress exceptions in parallel jobs")
        .def_rw("prim_tol", &ns_sqp::settings_t::prim_tol, "Primal feasibility tolerance")
        .def_rw("dual_tol", &ns_sqp::settings_t::dual_tol, "Dual feasibility tolerance")
        .def_rw("comp_tol", &ns_sqp::settings_t::comp_tol, "Complementarity feasibility tolerance")
        .def_rw("s_max", &ns_sqp::settings_t::s_max, "IPOPT-style dual scaling parameter: s_d = max(s_max, ||λ||_1/n_constr)/s_max");

    nb::class_<ns_sqp::scaling_settings> sc_setting(
        sqp, "scaling_settings",
        "Jacobian scaling mode and recomputation thresholds.");
    sc_setting
        .def_rw("scaling_mode", &ns_sqp::scaling_settings::mode,
                nb::for_getter(nb::sig("def scaling_mode(self) -> mode")),
                "Scaling mode: none, gradient (default), or equilibrium")
        .def_rw("equilibrium_iters", &ns_sqp::scaling_settings::equilibrium_iters,
                "Number of Ruiz iterations for equilibrium scaling")
        .def_rw("min_scale", &ns_sqp::scaling_settings::min_scale,
                "Minimum scale factor clamp (avoids division by zero)")
        .def_rw("update_ratio_threshold", &ns_sqp::scaling_settings::update_ratio_threshold,
                "Recompute scales when dual_res / prim_res >= this threshold");
    moto::export_enum<ns_sqp::scaling_settings::mode_t>(
        sc_setting, "Available Jacobian scaling algorithms.");

    nb::enum_<moto::solver::ipm_config::adaptive_mu_t> enum_binder(
        sqp, "adaptive_mu_t", "Barrier-parameter update strategies.");
    moto::export_enum<ns_sqp::iter_result_t>(
        sqp, "Termination status returned by an SQP update.");
    nb::class_<ns_sqp::profile_phase_stat>(
        sqp, "profile_phase_stat",
        "Aggregated wall-clock statistics for one solver phase.")
        .def_ro("name", &ns_sqp::profile_phase_stat::name, "Stable solver phase name")
        .def_ro("total_ms", &ns_sqp::profile_phase_stat::total_ms, "Total wall-clock milliseconds spent in this phase")
        .def_ro("avg_ms", &ns_sqp::profile_phase_stat::avg_ms, "Average milliseconds per recorded call")
        .def_ro("calls", &ns_sqp::profile_phase_stat::calls, "Number of recorded calls")
        .def_ro("share_of_update", &ns_sqp::profile_phase_stat::share_of_update, "Fraction of the complete update wall time");
    nb::class_<ns_sqp::profile_iteration>(
        sqp, "profile_iteration",
        "Wall-clock and trial-evaluation statistics for one SQP iteration.")
        .def_ro("index", &ns_sqp::profile_iteration::index, "One-based SQP iteration index")
        .def_ro("total_ms", &ns_sqp::profile_iteration::total_ms, "Total wall-clock milliseconds for the iteration")
        .def_ro("ls_steps", &ns_sqp::profile_iteration::ls_steps, "Number of line-search backtracking reductions")
        .def_ro("trial_evaluations", &ns_sqp::profile_iteration::trial_evaluations, "Number of nonlinear trial-point evaluations");
    nb::class_<ns_sqp::profile_report>(
        sqp, "profile_report",
        "Wall-clock profile collected by the most recent profiled update.")
        .def_ro("total_ms", &ns_sqp::profile_report::total_ms, "Total wall-clock milliseconds for the profiled update")
        .def_ro("initialize_ms", &ns_sqp::profile_report::initialize_ms, "Milliseconds spent initializing and linearizing the solve")
        .def_ro("sqp_iterations", &ns_sqp::profile_report::sqp_iterations, "Number of recorded SQP iterations")
        .def_ro("trial_evaluations", &ns_sqp::profile_report::trial_evaluations, "Total nonlinear trial-point evaluations")
        .def_ro("phases", &ns_sqp::profile_report::phases,
                nb::for_getter(nb::sig("def phases(self) -> list[profile_phase_stat]")),
                "Aggregated statistics for phases that were executed")
        .def_ro("iterations", &ns_sqp::profile_report::iterations,
                nb::for_getter(nb::sig("def iterations(self) -> list[profile_iteration]")),
                "Per-iteration wall-clock statistics");
    nb::class_<ns_sqp::iter_info>(
        sqp, "iter_info", "Termination status and iteration count.")
        .def_ro("result", &ns_sqp::iter_info::result,
                nb::for_getter(nb::sig("def result(self) -> iter_result")),
                "Result of the SQP iteration")
        .def_prop_ro("solved", [](const ns_sqp::iter_info &self) { return self.result == ns_sqp::iter_result_t::success; }, "Whether the problem is solved")
        .def_rw("num_iter", &ns_sqp::iter_info::num_iter, "Number of iterations");

    nb::class_<ns_sqp::kkt_info::barrier_objective_info>(
        sqp, "barrier_objective_info",
        "Cost, barrier, and line-search objective values at an iterate.")
        .def_ro("cost", &ns_sqp::kkt_info::barrier_objective_info::cost, "Original nonlinear objective value")
        .def_ro("barrier_value", &ns_sqp::kkt_info::barrier_objective_info::barrier_value, "Interior-point logarithmic barrier contribution")
        .def_ro("augmented_objective", &ns_sqp::kkt_info::barrier_objective_info::augmented_objective, "Objective plus active barrier and augmentation terms")
        .def_ro("ls_objective", &ns_sqp::kkt_info::barrier_objective_info::ls_objective, "Objective value used by globalization");

    nb::class_<ns_sqp::kkt_info::primal_info>(
        sqp, "primal_info",
        "Primal feasibility and complementarity residual summary.")
        .def_ro("inf_res", &ns_sqp::kkt_info::primal_info::inf_res, "Infinity norm of primal constraint violation")
        .def_ro("res_l1", &ns_sqp::kkt_info::primal_info::res_l1, "L1 norm of primal constraint violation")
        .def_ro("inf_comp", &ns_sqp::kkt_info::primal_info::inf_comp, "Infinity norm of complementarity residual");

    nb::class_<ns_sqp::kkt_info::dual_info>(
        sqp, "dual_info", "Dual stationarity and multiplier-norm summary.")
        .def_ro("inf_res", &ns_sqp::kkt_info::dual_info::inf_res, "Infinity norm of the Lagrangian stationarity residual")
        .def_ro("max_eq_norm", &ns_sqp::kkt_info::dual_info::max_eq_norm, "Largest hard-equality multiplier norm")
        .def_ro("max_ineq_norm", &ns_sqp::kkt_info::dual_info::max_ineq_norm, "Largest inequality or soft-constraint multiplier norm")
        .def_ro("max_norm", &ns_sqp::kkt_info::dual_info::max_norm, "Largest multiplier norm across all constraint types");

    nb::class_<ns_sqp::kkt_info::barrier_step_info>(
        sqp, "barrier_step_info",
        "Predicted barrier and line-search objective change for an SQP step.")
        .def_ro("search_barrier_dir_deriv", &ns_sqp::kkt_info::barrier_step_info::search_barrier_dir_deriv, "Directional derivative of the barrier search objective")
        .def_ro("augmented_objective_fullstep_dec", &ns_sqp::kkt_info::barrier_step_info::augmented_objective_fullstep_dec, "Predicted augmented-objective decrease for a full step")
        .def_ro("ls_objective_fullstep_dec", &ns_sqp::kkt_info::barrier_step_info::ls_objective_fullstep_dec, "Predicted globalization-objective decrease for a full step");

    nb::class_<ns_sqp::kkt_info::step_info>(
        sqp, "step_info", "Infinity norms of the latest primal and dual steps.")
        .def_ro("inf_prim_step", &ns_sqp::kkt_info::step_info::inf_prim_step, "Infinity norm of the primal step")
        .def_ro("inf_dual_step", &ns_sqp::kkt_info::step_info::inf_dual_step, "Infinity norm of the complete dual step")
        .def_ro("inf_eq_dual_step", &ns_sqp::kkt_info::step_info::inf_eq_dual_step, "Infinity norm of the hard-equality multiplier step")
        .def_ro("inf_ineq_dual_step", &ns_sqp::kkt_info::step_info::inf_ineq_dual_step, "Infinity norm of inequality and soft-constraint multiplier steps");

    nb::class_<ns_sqp::kkt_info>(
        sqp, "kkt_info", "KKT residual, objective, and step diagnostics.")
        .def_ro("barrier_objective", &ns_sqp::kkt_info::barrier_objective,
                nb::for_getter(nb::sig("def barrier_objective(self) -> barrier_objective_info")),
                "Objective and barrier values")
        .def_ro("primal", &ns_sqp::kkt_info::primal,
                nb::for_getter(nb::sig("def primal(self) -> primal_info")),
                "Primal feasibility and complementarity summary")
        .def_ro("dual", &ns_sqp::kkt_info::dual,
                nb::for_getter(nb::sig("def dual(self) -> dual_info")),
                "Dual stationarity and multiplier summary")
        .def_ro("barrier_step", &ns_sqp::kkt_info::barrier_step,
                nb::for_getter(nb::sig("def barrier_step(self) -> barrier_step_info")),
                "Predicted objective changes for the latest direction")
        .def_ro("step", &ns_sqp::kkt_info::step,
                nb::for_getter(nb::sig("def step(self) -> step_info")),
                "Primal and dual step norms");

    nb::class_<ns_sqp::result_type, ns_sqp::kkt_info>(
        sqp, "result_type",
        "Result returned by ``sqp.update``, including termination and KKT diagnostics.")
        .def_ro("iter", &ns_sqp::result_type::iter,
                nb::for_getter(nb::sig("def iter(self) -> iter_info")),
                "Iteration metadata")
        .def_prop_ro("result", [](const ns_sqp::result_type &self) { return self.iter.result; },
                     nb::sig("def result(self) -> iter_result"),
                     "Result of the SQP iteration")
        .def_prop_ro("solved", [](const ns_sqp::result_type &self) { return self.iter.result == ns_sqp::iter_result_t::success; }, "Whether the problem is solved")
        .def_prop_ro("num_iter", [](const ns_sqp::result_type &self) { return self.iter.num_iter; }, "Number of iterations")
        .def_prop_ro("inf_prim_res", [](const ns_sqp::result_type &self) { return self.primal.inf_res; },
                     "Convenience view of ``primal.inf_res``")
        .def_prop_ro("inf_dual_res", [](const ns_sqp::result_type &self) { return self.dual.inf_res; },
                     "Convenience view of ``dual.inf_res``")
        .def_prop_ro("inf_comp_res", [](const ns_sqp::result_type &self) { return self.primal.inf_comp; },
                     "Convenience view of ``primal.inf_comp``");

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<moto::solver::ipm_config::adaptive_mu_t>()) {
        enum_binder.value(std::string(name).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER

    nb::class_<ns_sqp::data, node_data>(
        sqp, "data_type",
        "Realized node data containing symbol values and function approximations.")
        .def_prop_ro(
            "prob", [](ns_sqp::data &self) -> auto & { return self.problem(); },
            nb::rv_policy::reference_internal,
            "Finalized OCP problem represented by this solver node")
        .def_prop_ro(
            "value", [](ns_sqp::data &self) -> auto & { return self.sym_val(); },
            nb::rv_policy::reference_internal,
            "Node-local symbol values")
        .def(
            "data",
            [](ns_sqp::data &self, const generic_func &function) -> auto & {
                return static_cast<node_data &>(self).data(function);
            },
            nb::arg("function"), nb::rv_policy::reference_internal,
            "Runtime value and derivative storage for a generated function");

    sqp.def_prop_ro("nodes",
                    [](ns_sqp &self) -> auto & { return self.solver_nodes(); },
                    nb::rv_policy::reference_internal,
                    nb::sig("def nodes(self) -> list[data_type]"),
                    "Ordered solver-node list");
}
