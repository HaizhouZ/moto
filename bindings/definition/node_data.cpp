#include <moto/ocp/impl/node_data.hpp>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>
#include <variant>

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    nb::class_<ocp_base>(m, "ocp_base")
        .def("add", [](ocp_base &self, expr_inarg_list &&exprs) { self.add(exprs); }, nb::arg("exprs"), "Add a list of expressions to the OCP problem")
        .def("add", [](ocp_base &self, shared_expr ex) { self.add(ex); }, nb::arg("ex"), "Add an expression to the OCP problem")
        .def("add_terminal", [](ocp_base &self, expr_inarg_list &&exprs) { self.add_terminal(exprs); }, nb::arg("exprs"), "Add a list of terminal expressions to the OCP problem")
        .def("add_terminal", [](ocp_base &self, shared_expr ex) { self.add_terminal(ex); }, nb::arg("ex"), "Add a terminal expression to the OCP problem")
        .def("dim", [](ocp_base &self, field_t field) { return self.dim(field); }, nb::arg("field"), "Get the dimension of the field")
        .def("wait_until_ready", &ocp_base::wait_until_ready, "Wait until all expressions in the OCP problem are ready")
        .def("is_active", &ocp_base::is_active, nb::arg("arg"), "Check if a given argument is active in the OCP problem")
        .def("print_summary", &ocp_base::print_summary, "Print a summary of the OCP problem");

    nb::class_<ocp, ocp_base>(m, "ocp");

    nb::class_<node_ocp, ocp>(m, "node_ocp")
        .def_static("create", &node_ocp::create, "Create a new node OCP problem")
        .def("clone", [](node_ocp &self) { return self.clone_node(); }, "Clone the node OCP problem")
        .def("clone", &node_ocp::clone_node, nb::arg("config"), "Clone the node OCP problem");

    nb::class_<edge_ocp, ocp>(m, "edge_ocp")
        .def_static("create", &edge_ocp::create, "Create a new edge OCP problem");

    nb::class_<ocp_base::active_status_config>(m, "active_status_config")
        .def(nb::init<>(), "Default constructor for active_status_config")
        .def(nb::init<expr_inarg_list, expr_inarg_list>(),
             nb::arg("deactivate_list") = nb::list{},
             nb::arg("activate_list") = nb::list{},
             "Constructor for active_status_config with deactivate and activate lists");

    nb::class_<sym_data>(m, "sym_data")
        .def("__getitem__", [](sym_data &self, py_var_inarg_wrapper s) -> auto { return self[s]; })
        .def("__setitem__", [](sym_data &self, py_var_inarg_wrapper s, std::variant<vector_ref, scalar_t> d) {
            if (std::holds_alternative<vector_ref>(d)) {
                self[s] = std::get<vector_ref>(d);
            } else if (std::holds_alternative<scalar_t>(d)) {
                assert(self[s].size() == 1 && "Cannot assign scalar to a vector variable");
                self[s](0) = std::get<scalar_t>(d);
            } else {
                throw std::runtime_error("Invalid type for sym_data assignment");
            } });

    nb::class_<node_data>(m, "node_data")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return self.problem(); }, nb::rv_policy::reference_internal)
        .def_prop_ro("value", [](node_data &self) -> auto & { return self.sym_val(); }, nb::rv_policy::reference_internal);

    nb::class_<func_approx_data>(m, "func_approx_data")
        .def("__getitem__", [](func_approx_data &self, py_var_inarg_wrapper s) { return self[(sym &)s]; })
        .def_prop_ro("v", [](func_approx_data &self) -> auto { return self.v_; }, "Value vector reference")
        .def(
            "jac",
            [](func_approx_data &self, py_var_inarg_wrapper in) -> auto { return self.jac((sym &)in); },
            "Get the jacobian reference for the input variable")
        .def("set_jac", [](func_approx_data &self, py_var_inarg_wrapper in, Eigen::Ref<const matrix> rhs) { self.jac((sym &)in) = rhs; });
}
