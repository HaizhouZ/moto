#include <moto/solver/ns_sqp.hpp>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <type_cast.hpp>

#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <enum_export.hpp>
using namespace moto;
using graph_type = ns_sqp::storage_type;
using binary_func_type = std::function<void(ns_sqp::data *, ns_sqp::data *)>;
using unary_func_type = std::function<void(ns_sqp::data *)>;

void register_submodule_ns_sqp(nb::module_ &m) {

    nb::class_<ns_sqp> sqp(m, "ns_sqp_impl");
    sqp.def(nb::init<size_t>(), "Constructor for the SQP solver with a specified number of jobs")
        .def_prop_ro("active_data", [](ns_sqp &self) -> graph_type & { return self.active_data(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("graph", [](ns_sqp &self) -> graph_model & { return self.graph(); }, nb::rv_policy::reference_internal)
        .def("create_graph", &ns_sqp::create_graph, nb::rv_policy::reference_internal, "Create a graph_model builder that synchronizes staged paths into this SQP solver")
        .def("update", [](ns_sqp &self, size_t n_iter, bool verbose) {
            nb::gil_scoped_release rel;
            return self.update(n_iter, verbose); }, nb::arg("n_iter") = 1, nb::arg("verbose") = true, "Update the SQP solver for a given number of iterations")
        .def("update_minimal", [](ns_sqp &self, size_t n_iter, bool verbose) {
            nb::gil_scoped_release rel;
            return self.update_minimal(n_iter, verbose); }, nb::arg("n_iter") = 1, nb::arg("verbose") = true, "Run a minimal baseline-like SQP loop that skips newer solver orchestration overhead")
        .def("reset_profile", &ns_sqp::reset_profile, "Clear the last SQP profile report")
        .def("get_profile_report", &ns_sqp::profile, "Get the latest SQP wall-clock profile report")
        .def_prop_ro("profile_report", &ns_sqp::profile, "Get the latest SQP wall-clock profile report")
        .def_ro("settings", &ns_sqp::settings, "Get the settings of the SQP solver");

    nb::class_<ns_sqp::ipm_config>(sqp, "ipm_config")
        .def_rw("mu0", &ns_sqp::ipm_config::mu0, "Initial barrier parameter for the IPM solver")
        .def_rw("warm_start", &ns_sqp::ipm_config::warm_start, "Whether to warm start the IPM solver")
        .def_rw("mu_method", &ns_sqp::ipm_config::mu_method, "Adaptive mu method for the IPM solver")
        .def_rw("mu_monotone_fraction_threshold", &ns_sqp::ipm_config::mu_monotone_fraction_threshold, "Threshold for monotone decrease of mu (smaller is more likely to use monotone decrease)")
        .def_rw("mu_monotone_factor", &ns_sqp::ipm_config::mu_monotone_factor, "Factor for monotone decrease of mu (smaller -> faster decrease)")
        .def_rw("globalization", &ns_sqp::ipm_config::globalization, "Whether to use globalization in the IPM solver");

    nb::class_<ns_sqp::iterative_refinement_setting> rf_setting(sqp, "iterative_refinement_setting");
    rf_setting.def_rw("enabled", &ns_sqp::iterative_refinement_setting::enabled, "Whether to use iterative refinement")
        .def_rw("max_iters", &ns_sqp::iterative_refinement_setting::max_iters, "Maximum number of iterative refinement iterations")
        .def_rw("prim_res_tol", &ns_sqp::iterative_refinement_setting::prim_res_tol, "Primal residual tolerance for iterative refinement")
        .def_rw("dual_res_tol", &ns_sqp::iterative_refinement_setting::dual_res_tol, "Dual residual tolerance for iterative refinement");

    nb::class_<ns_sqp::restoration_settings> restoration_setting(sqp, "restoration_settings");
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

    nb::class_<ns_sqp::equality_multiplier_init_settings> eq_init_setting(sqp, "equality_multiplier_init_settings");
    eq_init_setting
        .def_rw("enabled", &ns_sqp::equality_multiplier_init_settings::enabled,
                "Whether equality-type multipliers are rebuilt during initialization")
        .def_rw("rebuild_after_restoration_exit", &ns_sqp::equality_multiplier_init_settings::rebuild_after_restoration_exit,
                "Whether to rebuild equality-type multipliers after restoration exits successfully")
        .def_rw("rho_eq", &ns_sqp::equality_multiplier_init_settings::rho_eq,
                "PMM penalty used for equality-type constraints in the equality-init overlay")
        .def_rw("rf", &ns_sqp::equality_multiplier_init_settings::rf,
                "Dedicated iterative-refinement settings used only during equality-multiplier initialization");

    auto ls_config_base = nb::class_<solver::linesearch_config>(m, "linesearch_config");
    ls_config_base.def_rw("update_alpha_dual", &solver::linesearch_config::update_alpha_dual, "Whether to update the dual step size during line search")
        .def_rw("eq_dual_alpha_source", &solver::linesearch_config::eq_dual_alpha_source, "Source for dual step size for equality constraints")
        .def_rw("ineq_dual_alpha_source", &solver::linesearch_config::ineq_dual_alpha_source, "Source for dual step size for inequality constraints");

    moto::export_enum<solver::linesearch_config::dual_alpha_source>(ls_config_base);

    nb::class_<ns_sqp::linesearch_setting, solver::linesearch_config> ls_setting(sqp, "linesearch_setting");
    ls_setting.def_rw("enabled", &ns_sqp::linesearch_setting::enabled, "Whether to use line search")
        .def_rw("max_steps", &ns_sqp::linesearch_setting::max_steps, "Maximum number of line search steps")
        .def_rw("enable_soc", &ns_sqp::linesearch_setting::enable_soc, "Whether to try a second-order correction before backtracking")
        .def_rw("max_soc_iter", &ns_sqp::linesearch_setting::max_soc_iter, "Maximum number of second-order correction retries per SQP iteration")
        .def_rw("failure_strategy", &ns_sqp::linesearch_setting::failure_strategy, "Line search failure backup strategy")
        .def_rw("method", &ns_sqp::linesearch_setting::method, "Line search method: filter (default) or merit_backtracking")
        .def_rw("primal_gamma", &ns_sqp::linesearch_setting::primal_gamma, "Primal improvement requirement for the filter (higher is stricter)")
        .def_rw("dual_gamma", &ns_sqp::linesearch_setting::dual_gamma, "Objective improvement requirement for the filter (higher is stricter)")
        .def_rw("constr_vio_min_frac", &ns_sqp::linesearch_setting::constr_vio_min_frac, "Threshold for switching condition (fraction of initial primal residual)")
        .def_rw("armijo_dec_frac", &ns_sqp::linesearch_setting::armijo_dec_frac, "Sufficient decrease tolerance (eta in Armijo condition), smaller -> more strict decrease requirement")
        .def_rw("s_phi", &ns_sqp::linesearch_setting::s_phi, "IPOPT switching condition exponent on objective decrease (s_phi in IPOPT paper, Section 3.3)")
        .def_rw("s_theta", &ns_sqp::linesearch_setting::s_theta, "IPOPT switching condition exponent on constraint violation (s_theta in IPOPT paper, Section 3.3)")
        .def_rw("merit_sigma", &ns_sqp::linesearch_setting::merit_sigma, "Merit backtracking: weight on ||dual residual||^2 relative to ||constraint violation||^2 (default 1.0)")
        .def_rw("enable_flat_obj_accept", &ns_sqp::linesearch_setting::enable_flat_obj_accept, "Accept step when objective is flat, iterate is nearly feasible, and step is non-trivial")
        .def_rw("flat_obj_dec_tol", &ns_sqp::linesearch_setting::flat_obj_dec_tol, "Threshold on |fullstep_dec| below which the objective is considered flat")
        .def_rw("flat_obj_prim_tol", &ns_sqp::linesearch_setting::flat_obj_prim_tol, "Primal residual must be below this for flat-objective accept")
        .def_rw("flat_obj_step_tol", &ns_sqp::linesearch_setting::flat_obj_step_tol, "Step norm must exceed this for flat-objective accept (ensures non-trivial step)");

    ls_setting.def_rw("backtrack_scheme", &ns_sqp::linesearch_setting::backtrack_scheme, "Backtracking scheme: linspace (default) or geometric")
        .def_rw("backtrack_factor", &ns_sqp::linesearch_setting::backtrack_factor, "Geometric backtracking reduction factor (alpha *= factor each step, used when backtrack_scheme == geometric)");

    moto::export_enum<ns_sqp::linesearch_setting::failure_backup_strategy>(ls_setting);
    moto::export_enum<ns_sqp::linesearch_setting::backtrack_scheme_t>(sqp);
    moto::export_enum<ns_sqp::linesearch_setting::search_method>(sqp);
    nb::class_<ns_sqp::settings_t>(sqp, "settings_type")
        .def_ro("mu", &ns_sqp::settings_t::mu, "Barrier parameter for the IPM solver")
        .def_rw("ipm_conditional_corrector", &ns_sqp::settings_t::ipm_conditional_corrector, "Whether to use conditional corrector in the IPM solver")
        .def_prop_ro("ipm", [](ns_sqp::settings_t &self) -> auto & { return self.ipm; }, "IPM settings")
        .def_rw("rf", &ns_sqp::settings_t::rf, "Iterative refinement settings")
        .def_prop_ro("restoration", [](ns_sqp::settings_t &self) -> auto & { return self.restoration; }, "Restoration settings")
        .def_prop_ro("eq_init", [](ns_sqp::settings_t &self) -> auto & { return self.eq_init; }, "Equality multiplier initialization settings")
        .def_prop_ro("ls", [](ns_sqp::settings_t &self) -> auto & { return self.ls; }, "Line search settings")
        .def_rw("scaling", &ns_sqp::settings_t::scaling, "Jacobian scaling settings")
        .def_rw("no_except", &ns_sqp::settings_t::no_except, "Whether to suppress exceptions in parallel jobs")
        .def_rw("prim_tol", &ns_sqp::settings_t::prim_tol, "Primal feasibility tolerance")
        .def_rw("dual_tol", &ns_sqp::settings_t::dual_tol, "Dual feasibility tolerance")
        .def_rw("comp_tol", &ns_sqp::settings_t::comp_tol, "Complementarity feasibility tolerance")
        .def_rw("s_max", &ns_sqp::settings_t::s_max, "IPOPT-style dual scaling parameter: s_d = max(s_max, ||λ||_1/n_constr)/s_max");

    nb::class_<ns_sqp::scaling_settings> sc_setting(sqp, "scaling_settings");
    sc_setting
        .def_rw("scaling_mode", &ns_sqp::scaling_settings::mode,
                "Scaling mode: none, gradient (default), or equilibrium")
        .def_rw("equilibrium_iters", &ns_sqp::scaling_settings::equilibrium_iters,
                "Number of Ruiz iterations for equilibrium scaling")
        .def_rw("min_scale", &ns_sqp::scaling_settings::min_scale,
                "Minimum scale factor clamp (avoids division by zero)")
        .def_rw("update_ratio_threshold", &ns_sqp::scaling_settings::update_ratio_threshold,
                "Recompute scales when dual_res / prim_res >= this threshold");
    moto::export_enum<ns_sqp::scaling_settings::mode_t>(sc_setting);

    nb::enum_<moto::solver::ipm_config::adaptive_mu_t> enum_binder(sqp, "adaptive_mu_t");
    moto::export_enum<ns_sqp::iter_result_t>(sqp);
    nb::class_<ns_sqp::profile_phase_stat>(sqp, "profile_phase_stat")
        .def_ro("name", &ns_sqp::profile_phase_stat::name)
        .def_ro("total_ms", &ns_sqp::profile_phase_stat::total_ms)
        .def_ro("avg_ms", &ns_sqp::profile_phase_stat::avg_ms)
        .def_ro("calls", &ns_sqp::profile_phase_stat::calls)
        .def_ro("share_of_update", &ns_sqp::profile_phase_stat::share_of_update);
    nb::class_<ns_sqp::profile_iteration>(sqp, "profile_iteration")
        .def_ro("index", &ns_sqp::profile_iteration::index)
        .def_ro("total_ms", &ns_sqp::profile_iteration::total_ms)
        .def_ro("ls_steps", &ns_sqp::profile_iteration::ls_steps)
        .def_ro("trial_evaluations", &ns_sqp::profile_iteration::trial_evaluations);
    nb::class_<ns_sqp::profile_report>(sqp, "profile_report")
        .def_ro("total_ms", &ns_sqp::profile_report::total_ms)
        .def_ro("initialize_ms", &ns_sqp::profile_report::initialize_ms)
        .def_ro("sqp_iterations", &ns_sqp::profile_report::sqp_iterations)
        .def_ro("trial_evaluations", &ns_sqp::profile_report::trial_evaluations)
        .def_ro("phases", &ns_sqp::profile_report::phases)
        .def_ro("iterations", &ns_sqp::profile_report::iterations);
    nb::class_<ns_sqp::kkt_info>(sqp, "kkt_info")
        .def_ro("result", &ns_sqp::kkt_info::result, "Result of the SQP iteration")
        .def_prop_ro("solved", [](const ns_sqp::kkt_info &self) { return self.result == ns_sqp::iter_result_t::success; }, "Whether the problem is solved")
        .def_rw("num_iter", &ns_sqp::kkt_info::num_iter, "Number of iterations")
        .def_rw("ls_steps", &ns_sqp::kkt_info::ls_steps, "Line-search trial count used in this SQP iteration")
        .def_rw("objective", &ns_sqp::kkt_info::objective, "Objective value")
        .def_ro("cost", &ns_sqp::kkt_info::cost, "Pure running cost (sum of __cost terms, no barrier)")
        .def_ro("search_barrier_value", &ns_sqp::kkt_info::search_barrier_value, "Search-only barrier contribution for the active phase")
        .def_ro("search_barrier_dir_deriv", &ns_sqp::kkt_info::search_barrier_dir_deriv, "Directional derivative of the search-only barrier contribution")
        .def_rw("inf_prim_res", &ns_sqp::kkt_info::inf_prim_res, "Primal residual (constraint violation)")
        .def_rw("inf_dual_res", &ns_sqp::kkt_info::inf_dual_res, "Dual residual (stationary condition)")
        .def_rw("inf_comp_res", &ns_sqp::kkt_info::inf_comp_res, "Inequality complementarity residual")
        .def_rw("inf_prim_step", &ns_sqp::kkt_info::inf_prim_step, "Infinity norm of the primal step")
        .def_rw("inf_dual_step", &ns_sqp::kkt_info::inf_dual_step, "Infinity norm of the dual step");

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<moto::solver::ipm_config::adaptive_mu_t>()) {
        enum_binder.value(std::string(name).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER

    nb::class_<ns_sqp::node_type>(sqp, "node_type")
        .def_prop_ro("addr", [](ns_sqp::node_type &self) { return fmt::format("{:p}", static_cast<const void *>(self.data_.get())); }, "Get the data address associated with this node")
        .def_prop_ro("data", [](ns_sqp::node_type &self) { return std::optional<node_data *>(self.data_.get()); }, "Get the data associated with this node");

    nb::class_<ns_sqp::data, node_data>(sqp, "data_type")
        .def_prop_ro("addr", [](ns_sqp::data &self) { return fmt::format("{:p}", static_cast<const void *>(&self)); }, "Get the data address associated with this node")
        .def(nb::init<ocp_ptr_t>(), nb::arg("prob"), "Constructor for ns_sqp data with OCP problem");

    nb::class_<graph_type> active_data(sqp, "storage_type");
    active_data.def(nb::init<>())
        .def("reserve", &graph_type::reserve, nb::arg("stage_capacity"), "Reserve storage for a linear runtime chain")
        .def("add", &graph_type::add, nb::arg("node"), "Add a node to the graph and return a reference to it", nb::rv_policy::reference)
        .def("add_head", &graph_type::add_head, nb::arg("node"), nb::rv_policy::reference, "Add a node and set it as the head")
        .def("add_tail", &graph_type::add_tail, nb::arg("node"), nb::rv_policy::reference, "Add a node and set it as the tail")
        .def("set_head", &graph_type::set_head, nb::arg("node"), nb::rv_policy::reference)
        .def("set_tail", &graph_type::set_tail, nb::arg("node"), nb::rv_policy::reference)
        .def("connect",
             [](graph_type &self, ns_sqp::node_type &start, ns_sqp::node_type &to, size_t steps) {
                 self.connect(start, to, {steps, true, true});
             },
             nb::arg("start"), nb::arg("to"), nb::arg("steps") = 2,
             "Connect two existing nodes with a path of the requested length")
        .def("add_edge",
             [](graph_type &self, ns_sqp::node_type &start, ns_sqp::node_type &to, size_t steps) {
                 self.add_edge(start, to, steps, true, true);
             },
             nb::arg("start"), nb::arg("to"), nb::arg("steps") = 2,
             "Add an edge from one node to another with a given number of steps")
        .def("insert_after",
             [](graph_type &self, ns_sqp::node_type &start, ns_sqp::node_type next, size_t steps) -> ns_sqp::node_type & {
                 return self.insert_after(start, std::move(next), {steps, true, true});
             },
             nb::arg("start"), nb::arg("node"), nb::arg("steps") = 2,
             nb::rv_policy::reference, "Add a new node after the given node and connect it immediately")
        .def("add_path",
             [](graph_type &self,
                std::vector<ns_sqp::node_type> nodes,
                const std::vector<size_t> &steps,
                bool set_head,
                bool set_tail) {
                 if (nodes.empty()) {
                     return std::vector<ns_sqp::node_type *>{};
                 }
                 if (steps.size() + 1 != nodes.size()) {
                     throw std::invalid_argument("graph.add_path expects exactly one fewer edge-length than nodes");
                 }
                 std::vector<ns_sqp::node_type *> added;
                 added.reserve(nodes.size());
                 for (auto &node : nodes) {
                     auto &added_node = self.add(std::move(node));
                     added.push_back(&added_node);
                 }
                 if (set_head) {
                     self.set_head(*added.front());
                 }
                 if (set_tail) {
                     self.set_tail(*added.back());
                 }
                 for (size_t i = 1; i < added.size(); ++i) {
                     self.connect(*added[i - 1], *added[i], {steps[i - 1], true, false});
                 }
                 return added;
             },
             nb::arg("nodes"),
             nb::arg("steps"),
             nb::arg("set_head") = false,
             nb::arg("set_tail") = false,
             "Add a sequence of nodes and connect adjacent pairs with the provided path lengths")
        .def("flatten_nodes", &graph_type::flatten_nodes, nb::rv_policy::reference, "Get the ordered flattened list of all nodes in the graph")
        .def_prop_ro("nodes", [](graph_type &self) -> auto & { return self.nodes(); }, nb::rv_policy::reference, "Linear runtime nodes");
}
