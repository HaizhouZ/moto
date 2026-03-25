#include <moto/solver/ns_sqp.hpp>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <type_cast.hpp>

#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>

#include <enum_export.hpp>
using namespace moto;
using graph_type = decltype(ns_sqp::graph_);
using binary_func_type = std::function<void(ns_sqp::data *, ns_sqp::data *)>;
using unary_func_type = std::function<void(ns_sqp::data *)>;
using unary_list = graph_type::unary_view::base;
using binary_list = graph_type::binary_view::base;
NB_MAKE_OPAQUE(unary_list);
NB_MAKE_OPAQUE(binary_list);

void register_submodule_ns_sqp(nb::module_ &m) {

    nb::class_<ns_sqp> sqp(m, "ns_sqp_impl");
    sqp.def(nb::init<size_t>(), "Constructor for the SQP solver with a specified number of jobs")
        .def_prop_ro("graph", [](ns_sqp &self) -> auto & { return self.graph_; })
        .def("update", [](ns_sqp &self, size_t n_iter, bool verbose) {
            nb::gil_scoped_release rel;
            return self.update(n_iter, verbose); }, nb::arg("n_iter") = 1, nb::arg("verbose") = true, "Update the SQP solver for a given number of iterations")
        .def_ro("settings", &ns_sqp::settings, "Get the settings of the SQP solver")
        .def("create_node", &ns_sqp::create_node, nb::arg("formulation"), nb::rv_policy::reference, "Create a new node in the SQP graph with the given OCP problem formulation");

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

    nb::class_<ns_sqp::linesearch_setting> ls_setting(sqp, "linesearch_setting");
    ls_setting.def_rw("enabled", &ns_sqp::linesearch_setting::enabled, "Whether to use line search")
        .def_rw("max_steps", &ns_sqp::linesearch_setting::max_steps, "Maximum number of line search steps")
        .def_rw("failure_backup", &ns_sqp::linesearch_setting::failure_backup, "Line search failure backup strategy")
        .def_rw("primal_gamma", &ns_sqp::linesearch_setting::primal_gamma, "Primal improvement requirement for the filter (higher is stricter)")
        .def_rw("dual_gamma", &ns_sqp::linesearch_setting::dual_gamma, "Dual improvement requirement for the filter (higher is stricter)")
        .def_rw("enable_dual_cut", &ns_sqp::linesearch_setting::enable_dual_cut, "Whether to enable the strict cut for dual residual when primal residual is small")
        .def_rw("eta", &ns_sqp::linesearch_setting::eta, "Elasticity coefficient for the dual cut when primal residual is small, used to relax the dual cut as line search step increases")
        .def_rw("dual_cut_coeff", &ns_sqp::linesearch_setting::dual_cut_coeff, "Cut threshold for dual residual when primal residual is small (higher is looser)");
    moto::export_enum<ns_sqp::linesearch_setting::failure_backup_strategy>(ls_setting);

    nb::class_<ns_sqp::settings_t>(sqp, "settings_type")
        .def_ro("mu", &ns_sqp::settings_t::mu, "Barrier parameter for the IPM solver")
        .def_rw("ipm_conditional_corrector", &ns_sqp::settings_t::ipm_conditional_corrector, "Whether to use conditional corrector in the IPM solver")
        .def_prop_ro("ipm", [](ns_sqp::settings_t &self) -> auto & { return self.ipm; }, "IPM settings")
        .def_rw("rf", &ns_sqp::settings_t::rf, "Iterative refinement settings")
        .def_prop_ro("ls", [](ns_sqp::settings_t &self) -> auto & { return self.ls; }, "Line search settings")
        .def_rw("no_except", &ns_sqp::settings_t::no_except, "Whether to suppress exceptions in parallel jobs")
        .def_rw("prim_tol", &ns_sqp::settings_t::prim_tol, "Primal feasibility tolerance")
        .def_rw("dual_tol", &ns_sqp::settings_t::dual_tol, "Dual feasibility tolerance")
        .def_rw("comp_tol", &ns_sqp::settings_t::comp_tol, "Complementarity feasibility tolerance");

    nb::enum_<moto::solver::ipm_config::adaptive_mu_t> enum_binder(sqp, "adaptive_mu_t");

    nb::class_<ns_sqp::kkt_info>(sqp, "kkt_info")
        .def_rw("solved", &ns_sqp::kkt_info::solved, "Whether the problem is solved")
        .def_rw("num_iter", &ns_sqp::kkt_info::num_iter, "Number of iterations")
        .def_rw("objective", &ns_sqp::kkt_info::objective, "Objective value")
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
        .def_prop_ro("addr", [](ns_sqp::node_type &self) { return fmt::format("{:p}", static_cast<const void *>(self.data_)); }, "Get the data address associated with this node")
        .def_prop_ro("data", [](ns_sqp::node_type &self) { return std::optional<node_data *>(self.data_); }, "Get the data associated with this node");

    nb::class_<ns_sqp::data, node_data>(sqp, "data_type")
        .def_prop_ro("addr", [](ns_sqp::data &self) { return fmt::format("{:p}", static_cast<const void *>(&self)); }, "Get the data address associated with this node")
        .def(nb::init<ocp_ptr_t>(), nb::arg("prob"), "Constructor for ns_sqp data with OCP problem");

    nb::class_<graph_type> graph(sqp, "graph_type");
    graph.def(nb::init<>())
        .def("add", &graph_type::add, nb::arg("node"), "Add a node to the graph and return a reference to it", nb::rv_policy::reference)
        .def("set_head", &graph_type::set_head, nb::arg("node"), nb::rv_policy::reference)
        .def("set_tail", &graph_type::set_tail, nb::arg("node"), nb::rv_policy::reference)
        .def("add_edge", &graph_type::add_edge, nb::arg("start"), nb::arg("to"), nb::arg("steps") = 1, nb::arg("include_st") = true, nb::arg("include_ed") = true,
             "Add an edge from one node to another with a given number of steps")
        .def("flatten_nodes", &graph_type::flatten_nodes, nb::rv_policy::reference, "Get the unordered flattened list of all nodes in the graph")
        .def_prop_ro("nodes", &graph_type::nodes, nb::rv_policy::reference, "key nodes")
        .def("forward_view", &graph_type::forward_view, nb::rv_policy::reference, "Get a forward view of the graph, i.e., a view of all nodes in forward direction")
        .def("backward_view", &graph_type::backward_view, nb::rv_policy::reference, "Get a backward view of the graph, i.e., a view of all nodes in backward direction")
        .def("forward_binary_view", &graph_type::forward_binary_view, nb::rv_policy::reference, nb::arg("none_on_end") = false, "Get a forward binary view of the graph, i.e., a view of all nodes in forward direction with pairs of nodes, last is none")
        .def("backward_binary_view", &graph_type::backward_binary_view, nb::rv_policy::reference, nb::arg("none_on_end") = false, "Get a backward binary view of the graph, i.e., a view of all nodes in backward direction with pairs of nodes, first is none");

    nb::bind_vector<unary_list>(graph, "unary_list");
    nb::class_<graph_type::unary_view, unary_list>(graph, "unary_view")
        .def("update", &graph_type::unary_view::update, "Update the unary view and return true if there are more nodes to process");

    nb::bind_vector<binary_list>(graph, "binary_list");
    nb::class_<graph_type::binary_view, binary_list>(graph, "binary_view")
        .def("update", &graph_type::binary_view::update, "Update the binary view and return true if there are more nodes to process");
}