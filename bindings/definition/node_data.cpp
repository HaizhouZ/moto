#include <moto/ocp/impl/data_mgr.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <type_cast.hpp>

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    nb::class_<ocp>(m, "ocp")
        .def("add", [](ocp &self, expr_inarg_list &&exprs) { self.add(exprs); }, nb::arg("exprs"), "Add a list of expressions to the OCP problem")
        .def("add", [](ocp &self, const py_shared_expr_wrapper &ex) { self.add((shared_expr &)ex); }, nb::arg("ex"), "Add an expression to the OCP problem")
        .def_static("create", &ocp::create, "Create a new OCP problem")
        .def("clone", &ocp::clone, "Clone the OCP problem")
        .def("dim", [](ocp &self, field_t field) { return self.dim(field); }, nb::arg("field"), "Get the dimension of the field")
        //    .def("exprs", [](ocp &self, field_t field) { return expr_inarg_list(self.exprs<shared_expr>(field)); }, nb::arg("field"), "Get the expressions in the field")
        .def("exprs", [](ocp &self, field_t field) -> auto & { return static_cast<const std::vector<shared_expr> &>(self.exprs<shared_expr>(field)); }, nb::arg("field"), "Get the expressions in the field", nb::rv_policy::reference_internal)
        .def_prop_ro("uid", &ocp::uid, "Get the unique identifier of the OCP problem");

    nb::class_<sym_data>(m, "sym_data")
        .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for sym_data with OCP problem")
        .def_prop_ro("prob", [](sym_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](sym_data &self, const py_var_wrapper &s) -> auto { return self[s]; })
        .def("__setitem__", [](sym_data &self, const py_var_wrapper &s, vector_ref d) { self[s] = d; });
    nb::class_<dense_approx_data>(m, "dense_approx_data")
        .def(nb::init<ocp *>(), nb::arg("prob"), "Constructor for dense_approx_data with OCP problem")
        .def_prop_ro("prob", [](dense_approx_data &self) -> ocp & { return *self.prob_; })
        .def_rw("cost", &dense_approx_data::cost_)
        .def_rw("approx", &dense_approx_data::approx_)
        .def_rw("dual", &dense_approx_data::dual_)
        .def_rw("comp", &dense_approx_data::comp_)
        .def_rw("hessian", &dense_approx_data::hessian_)
        .def_rw("jac", &dense_approx_data::jac_)
        .def_rw("jac_modification", &dense_approx_data::jac_modification_);
    nb::class_<shared_data>(m, "shared_data")
        .def(nb::init<const ocp *, sym_data *>(), nb::arg("prob"), nb::arg("primal"),
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
        .def("data", &node_data::data, nb::arg("func"), "Get the sparse func data by pointer")
        .def("cost", &node_data::cost, "Get the cost value");
    nb::class_<func_arg_map>(m, "func_arg_map")
        .def_prop_ro("prob", [](func_arg_map &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_arg_map &self, const py_var_wrapper &s) { return self[s]; })
        .def("__setitem__", [](func_arg_map &self, const py_var_wrapper &s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with sym_data and shared_data")
        .def(nb::init<std::vector<vector_ref> &&, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with a list of primal data and shared_data");
    nb::class_<func_approx_map, func_arg_map>(m, "func_approx_map")
        .def_prop_ro("prob", [](func_approx_map &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_approx_map &self, const py_var_wrapper &s) { return self[s]; })
        .def("__setitem__", [](func_approx_map &self, const py_var_wrapper &s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, dense_approx_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("raw"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_map with sym_data, dense_approx_data and shared_data")
        .def(nb::init<sym_data &, vector_ref, std::vector<matrix_ref> &&, shared_data &, const func &>(),
             nb::arg("primal"), nb::arg("v"), nb::arg("jac"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_map with sym_data, vector_ref to input, list of matrix_ref to jacobian and shared_data")
        .def("setup_hessian", &func_approx_map::setup_hessian, nb::arg("raw"),
             "Setup hessian from raw approximation data")
        .def_prop_ro("v", [](func_approx_map &self) -> auto { return self.v_; }, "Value vector reference")
        .def("jac", [](func_approx_map &self, size_t idx) -> auto { return self.jac_[idx]; }, "Jacobian matrix references indexed by input arguments")
        .def_prop_ro("hess", [](func_approx_map &self) -> auto & { return self.hess_; }, "Hessian matrix references for merit, 2-D indexed by input arguments")
        .def("jac", [](func_approx_map &self, const py_var_wrapper &in) -> auto { return self.jac(in); }, nb::arg("in"), "Get the jacobian reference for the input variable");
}