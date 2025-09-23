#include <moto/ocp/impl/data_mgr.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>
#include <variant>

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    auto ocp_handle =
        nb::class_<ocp>(m, "ocp")
            .def("add", [](ocp &self, expr_inarg_list &&exprs) { self.add(exprs); }, nb::arg("exprs"), "Add a list of expressions to the OCP problem")
            .def("add", [](ocp &self, const py_shared_expr_wrapper &ex) { self.add((shared_expr &)ex); }, nb::arg("ex"), "Add an expression to the OCP problem")
            .def_static("create", &ocp::create, "Create a new OCP problem")
            .def("clone", [](ocp &self, ocp::clone_config &&config) { return self.clone(config); }, "Clone the OCP problem")
            .def("clone", [](ocp &self) { return self.clone(); }, "Clone the OCP problem")
            .def("dim", [](ocp &self, field_t field) { return self.dim(field); }, nb::arg("field"), "Get the dimension of the field")
            //    .def("exprs", [](ocp &self, field_t field) { return expr_inarg_list(self.exprs<shared_expr>(field)); }, nb::arg("field"), "Get the expressions in the field")
            .def("exprs", [](ocp &self, field_t field) -> auto & { return static_cast<const std::vector<shared_expr> &>(self.exprs(field)); }, nb::arg("field"), "Get the expressions in the field", nb::rv_policy::reference_internal)
            .def_prop_ro("uid", &ocp::uid, "Get the unique identifier of the OCP problem")
            .def("wait_until_ready", &ocp::wait_until_ready, "Wait until all expressions in the OCP problem are ready")
            .def("print_summary", &ocp::print_summary, "Print a summary of the OCP problem");

    nb::class_<ocp::clone_config>(ocp_handle, "clone_config")
        .def(nb::init<>(), "Default constructor for clone_config")
        .def(nb::init<expr_inarg_list, expr_inarg_list>(),
             nb::arg("deactivate_list") = nb::list{},
             nb::arg("activate_list") = nb::list{},
             "Constructor for clone_config with deactivate and activate lists");

    nb::class_<sym_data>(m, "sym_data")
        .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for sym_data with OCP problem")
        .def_prop_ro("prob", [](sym_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](sym_data &self, const py_var_wrapper &s) -> auto { return self[s]; })
        .def("print", &sym_data::print, "Print the symbolic data")
        .def("__setitem__", [](sym_data &self, const py_var_wrapper &s, std::variant<vector_ref, scalar_t> d) { 
            if (std::holds_alternative<vector_ref>(d)) {
                self[s] = std::get<vector_ref>(d);
            } else if (std::holds_alternative<scalar_t>(d)) {
                assert(self[s].size() == 1 && "Cannot assign scalar to a vector variable");
                self[s](0) = std::get<scalar_t>(d);
            } else {
                throw std::runtime_error("Invalid type for sym_data assignment");
            } });

    nb::class_<merit_data>(m, "merit_data")
        .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for merit_data with OCP problem")
        .def_prop_ro("prob", [](merit_data &self) -> ocp & { return *self.prob_; })
        .def_rw("cost", &merit_data::cost_)
        .def_rw("approx", &merit_data::approx_)
        .def_rw("dual", &merit_data::dual_)
        .def_rw("comp", &merit_data::comp_)
        .def_rw("hessian", &merit_data::hessian_)
        .def_rw("jac", &merit_data::jac_)
        .def_rw("jac_modification", &merit_data::jac_modification_);

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
        .def("__getitem__", [](func_arg_map &self, const py_var_wrapper &s) { return self[s]; })
        .def("__setitem__", [](func_arg_map &self, const py_var_wrapper &s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with sym_data and shared_data");

    nb::class_<func_approx_data, func_arg_map>(m, "func_approx_data")
        .def_prop_ro("prob", [](func_approx_data &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_approx_data &self, const py_var_wrapper &s) { return self[s]; })
        .def("__setitem__", [](func_approx_data &self, const py_var_wrapper &s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, merit_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("raw"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_data with sym_data, merit_data and shared_data")
        .def("setup_hessian", &func_approx_data::setup_hessian, "Setup hessian from raw approximation data")
        .def_rw("v", &func_approx_data::v_, "Value vector reference")
        // .def_rw("jac", &func_approx_data::jac_, "Jacobian matrix references indexed by input arguments")
        // .def_rw("hess", &func_approx_data::hess_, "Hessian matrix references for merit, 2-D indexed by input arguments")
        .def(
            "jac",
            [](func_approx_data &self, const py_var_wrapper &in) -> auto { return self.jac(in); },
            "Get the jacobian reference for the input variable")
        .def("set_jac", [](func_approx_data &self, const py_var_wrapper &in, Eigen::Ref<const matrix> rhs) { self.jac(in) = rhs; });
}