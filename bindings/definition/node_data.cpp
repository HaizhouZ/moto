#include <moto/ocp/impl/node_data.hpp>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>
#include <variant>

namespace {
nb::list to_py_list(const moto::var_list &items) {
    nb::list out;
    for (const auto &item : items) {
        out.append(nb::cast(item));
    }
    return out;
}
} // namespace

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    auto ocp_base_handle =
        nb::class_<ocp_base>(m, "ocp_base")
            .def("add", [](ocp_base &self, expr_inarg_list &&exprs) { self.add(exprs); }, nb::arg("exprs"), "Add a list of expressions to the OCP problem")
            .def("add", [](ocp_base &self, shared_expr ex) { self.add(ex); }, nb::arg("ex"), "Add an expression to the OCP problem")
            .def("add_terminal", [](ocp_base &self, expr_inarg_list &&exprs) { self.add_terminal(exprs); }, nb::arg("exprs"), "Add a list of terminal expressions to the OCP problem")
            .def("add_terminal", [](ocp_base &self, shared_expr ex) { self.add_terminal(ex); }, nb::arg("ex"), "Add a terminal expression to the OCP problem")
            .def("dim", [](ocp_base &self, field_t field) { return self.dim(field); }, nb::arg("field"), "Get the dimension of the field")
            .def("exprs", [](ocp_base &self, field_t field) -> const auto & { return static_cast<const std::vector<shared_expr> &>(self.exprs(field)); }, nb::arg("field"), "Get the expressions in the field", nb::rv_policy::reference_internal)
            .def_prop_ro("uid", &ocp_base::uid, "Get the unique identifier of the OCP problem")
            .def("wait_until_ready", &ocp_base::wait_until_ready, "Wait until all expressions in the OCP problem are ready")
            .def("update_active_status", &ocp_base::update_active_status, nb::arg("config"), nb::arg("update_sub_probs") = true, "Update the active status of the OCP problem based on the provided configuration")
            .def("is_active", &ocp_base::is_active, nb::arg("arg"), nb::arg("include_sub_prob") = true, "Check if a given argument is active in the OCP problem")
            .def("print_summary", &ocp_base::print_summary, "Print a summary of the OCP problem");

    auto ocp_handle =
        nb::class_<ocp, ocp_base>(m, "ocp")
            .def_static("create", &ocp::create, "Create a new OCP problem")
            .def("clone", [](ocp &self, ocp::active_status_config &&config) { return self.clone(config); }, "Clone the OCP problem")
            .def("clone", [](ocp &self) { return self.clone(); }, "Clone the OCP problem");

    nb::class_<node_ocp, ocp>(m, "node_ocp")
        .def_static("create", &node_ocp::create, "Create a new node OCP problem")
        .def("clone", [](node_ocp &self, node_ocp::active_status_config &&config) { return self.clone_node(config); }, "Clone the node OCP problem")
        .def("clone", [](node_ocp &self) { return self.clone_node(); }, "Clone the node OCP problem");

    nb::class_<edge_ocp, ocp>(m, "edge_ocp")
        .def_static("create", &edge_ocp::create, "Create a new edge OCP problem")
        .def("clone", [](edge_ocp &self, edge_ocp::active_status_config &&config) { return self.clone_edge(config); }, "Clone the edge OCP problem")
        .def("clone", [](edge_ocp &self) { return self.clone_edge(); }, "Clone the edge OCP problem")
        .def_prop_ro("st_prob", &edge_ocp::st_node_prob, "Get the edge source node problem")
        .def_prop_ro("ed_prob", &edge_ocp::ed_node_prob, "Get the edge successor node problem");

    nb::class_<ocp_base::active_status_config>(ocp_handle, "active_status_config")
        .def(nb::init<>(), "Default constructor for active_status_config")
        .def(nb::init<expr_inarg_list, expr_inarg_list>(),
             nb::arg("deactivate_list") = nb::list{},
             nb::arg("activate_list") = nb::list{},
             "Constructor for active_status_config with deactivate and activate lists");

    nb::class_<sym_data>(m, "sym_data")
        .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for sym_data with OCP problem")
        .def_prop_ro("prob", [](sym_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](sym_data &self, py_var_inarg_wrapper s) -> auto { return self[s]; })
        .def("print", &sym_data::print, "Print the symbolic data")
        .def("__setitem__", [](sym_data &self, py_var_inarg_wrapper s, std::variant<vector_ref, scalar_t> d) {
            if (std::holds_alternative<vector_ref>(d)) {
                self[s] = std::get<vector_ref>(d);
            } else if (std::holds_alternative<scalar_t>(d)) {
                assert(self[s].size() == 1 && "Cannot assign scalar to a vector variable");
                self[s](0) = std::get<scalar_t>(d);
            } else {
                throw std::runtime_error("Invalid type for sym_data assignment");
            } });

    auto lag_data_handle =
        nb::class_<lag_data>(m, "lag_data")
            .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for lag_data with OCP problem")
            .def_prop_ro("prob", [](lag_data &self) -> ocp & { return *self.prob_; })
            .def_rw("lag", &lag_data::lag_)
            .def_rw("cost", &lag_data::cost_)
            .def_rw("approx", &lag_data::approx_)
            .def_rw("dual", &lag_data::dual_)
            .def_rw("comp", &lag_data::comp_)
            .def_rw("lag_hess", &lag_data::lag_hess_)
            .def_rw("lag_jac", &lag_data::lag_jac_)
            .def_rw("lag_jac_corr", &lag_data::lag_jac_corr_);
    m.attr("lag_data") = lag_data_handle;

    nb::class_<shared_data>(m, "shared_data")
        .def(nb::init<ocp *, sym_data *>(), nb::arg("prob"), nb::arg("primal"),
             "Constructor for shared data with OCP problem and sym data")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return self.problem(); })
        .def("__getitem__", [](shared_data &self, const generic_func &f) -> auto & { return self[f]; });

    nb::class_<node_data>(m, "node_data")
        .def(nb::init<ocp_ptr_t>(), nb::arg("prob"), "Constructor for node_data with OCP problem")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return self.problem(); })
        .def_prop_ro("sym", [](node_data &self) -> auto & { return self.sym_val(); })
        .def_prop_ro("value", [](node_data &self) -> auto & { return self.sym_val(); })
        .def_prop_ro("dense", [](node_data &self) -> auto & { return self.dense(); })
        .def_prop_ro("shared", [](node_data &self) -> auto & { return self.shared(); })
        .def("print_residuals", &node_data::print_residuals, "Print the residuals of the node data")
        .def("data", nb::overload_cast<const func &>(&node_data::data, nb::const_), nb::arg("func"), "Get the sparse func data by pointer")
        .def("data", nb::overload_cast<const custom_func &>(&node_data::data, nb::const_), nb::arg("func"), "Get the custom func data by pointer")
        .def("cost", &node_data::cost, "Get the cost value");

    nb::class_<func_arg_map>(m, "func_arg_map")
        .def_prop_ro("prob", [](func_arg_map &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_arg_map &self, py_var_inarg_wrapper s) { return self[(sym &)s]; })
        .def("__setitem__", [](func_arg_map &self, py_var_inarg_wrapper s, vector_ref d) { self[(sym &)s] = d; })
        .def(nb::init<sym_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with sym_data and shared_data");

    nb::class_<func_approx_data, func_arg_map>(m, "func_approx_data")
        .def_prop_ro("prob", [](func_approx_data &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_approx_data &self, py_var_inarg_wrapper s) { return self[(sym &)s]; })
        .def("__setitem__", [](func_approx_data &self, py_var_inarg_wrapper s, vector_ref d) { self[(sym &)s] = d; })
        .def(nb::init<sym_data &, lag_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("raw"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_data with sym_data, lag_data and shared_data")
        .def("setup_hessian", &func_approx_data::setup_hessian, "Setup hessian from raw approximation data")
        .def_rw("v", &func_approx_data::v_, "Value vector reference")
        // .def_rw("jac", &func_approx_data::jac_, "Jacobian matrix references indexed by input arguments")
        // .def_rw("hess", &func_approx_data::hess_, "Hessian matrix references for merit, 2-D indexed by input arguments")
        .def(
            "jac",
            [](func_approx_data &self, py_var_inarg_wrapper in) -> auto { return self.jac((sym &)in); },
            "Get the jacobian reference for the input variable")
        .def("set_jac", [](func_approx_data &self, py_var_inarg_wrapper in, Eigen::Ref<const matrix> rhs) { self.jac((sym &)in) = rhs; });
}
