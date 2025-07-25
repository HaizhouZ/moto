#include <moto/ocp/impl/node_data.hpp>
#include <type_cast.hpp>
#include <definition/sym_type_caster.hpp>

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    nb::class_<ocp>(m, "ocp")
        .def("add", &ocp::add<expr>, nb::arg("ex"), "Add an expression to the OCP problem")
        .def("add", [](ocp &self, sym_in_list &&exprs) { self.add(exprs); }, nb::arg("exprs"), "Add a list of expressions to the OCP problem")
        .def_static("create", &ocp::create, "Create a new OCP problem")
        .def("clone", &ocp::clone, "Clone the OCP problem")
        .def("dim", &ocp::dim, nb::arg("field"), "Get the dimension of the field")
        .def("exprs", &ocp::exprs<expr>, nb::arg("field"), "Get the expressions in the field")
        .def_prop_ro("uid", &ocp::uid, "Get the unique identifier of the OCP problem");

    nb::class_<sym_data>(m, "sym_data")
        .def_prop_ro("prob", [](sym_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](sym_data &self, sym *s) -> auto { return self[s]; });
    nb::class_<dense_approx_data>(m, "dense_approx_data")
        .def_prop_ro("prob", [](dense_approx_data &self) -> ocp & { return *self.prob_; })
        .def_rw("cost", &dense_approx_data::cost_)
        .def_rw("approx", &dense_approx_data::approx_)
        .def_rw("dual", &dense_approx_data::dual_)
        .def_rw("comp", &dense_approx_data::comp_)
        .def_rw("hessian", &dense_approx_data::hessian_)
        .def_rw("jac", &dense_approx_data::jac_)
        .def_rw("jac_modification", &dense_approx_data::jac_modification_);
    nb::class_<shared_data>(m, "shared_data")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return self.problem(); })
        .def("__getitem__", [](shared_data &self, const func &f) -> auto & { return self[f]; });
    nb::class_<node_data>(m, "node_data")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return self.problem(); })
        .def_prop_ro("sym", [](node_data &self) -> auto & { return self.sym_val(); })
        .def_prop_ro("dense", [](node_data &self) -> auto & { return self.dense(); })
        .def_prop_ro("shared", [](node_data &self) -> auto & { return self.shared(); })
        .def("value", nb::overload_cast<field_t>(&node_data::value, nb::const_), nb::arg("field"),
             "Get value of the whole field")
        .def("value", nb::overload_cast<sym *>(&node_data::value, nb::const_), nb::arg("sym"),
             "Get value of the sym variable")
        .def("data", &node_data::data, nb::arg("func"),
             "Get the sparse func data by pointer")
        .def("cost", &node_data::cost, "Get the cost value");
    nb::class_<func_arg_map>(m, "func_arg_map")
        .def_prop_ro("prob", [](func_arg_map &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_arg_map &self, sym *s) { return self[s]; })
        .def("__setitem__", [](func_arg_map &self, sym *s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with sym_data and shared_data")
        .def(nb::init<std::vector<vector_ref> &&, shared_data &, const func &>(), nb::arg("primal"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_arg_map with a list of primal data and shared_data");
    nb::class_<func_approx_map, func_arg_map>(m, "func_approx_map")
        .def_prop_ro("prob", [](func_approx_map &self) -> auto & { return *self.problem(); })
        .def("__getitem__", [](func_approx_map &self, sym *s) { return self[s]; })
        .def("__setitem__", [](func_approx_map &self, sym *s, vector_ref d) { self[s] = d; })
        .def(nb::init<sym_data &, dense_approx_data &, shared_data &, const func &>(), nb::arg("primal"), nb::arg("raw"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_map with sym_data, dense_approx_data and shared_data")
        .def(nb::init<sym_data &, vector_ref, std::vector<matrix_ref> &&, shared_data &, const func &>(),
             nb::arg("primal"), nb::arg("v"), nb::arg("jac"), nb::arg("shared"), nb::arg("f"),
             "Constructor for func_approx_map with sym_data, vector_ref to input, list of matrix_ref to jacobian and shared_data")
        .def("setup_hessian", &func_approx_map::setup_hessian, nb::arg("raw"),
             "Setup hessian from raw approximation data")
        .def("jac", nb::overload_cast<sym *>(&func_approx_map::jac, nb::const_), nb::arg("in"),
             "Get the jacobian reference for the input symbol")
        .def_prop_ro("v", [](func_approx_map &self) -> auto { return self.v_; }, "Value vector reference")
        .def_prop_ro("jac", [](func_approx_map &self) -> auto & { return self.jac_; }, "Jacobian matrix references indexed by input arguments")
        .def_prop_ro("hess", [](func_approx_map &self) -> auto & { return self.hess_; }, "Hessian matrix references for merit, 2-D indexed by input arguments");
}