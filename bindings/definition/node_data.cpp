#include <moto/ocp/impl/node_data.hpp>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>
#include <variant>

namespace {

moto::ocp_base::active_status_config active_status(
    const moto::expr_inarg_list &disable,
    const moto::expr_inarg_list &enable) {
    moto::ocp_base::active_status_config result;
    result.deactivate_list = moto::expr_list(disable);
    result.activate_list = moto::expr_list(enable);
    return result;
}

} // namespace

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    nb::class_<ocp_base>(
        m, "ocp_base",
        "Finalizable collection of active model expressions and their field layout.")
        .def("add", [](ocp_base &self, expr_inarg_list &&exprs) { self.add(exprs); },
             nb::arg("exprs"), "Add expressions by shared handle; this does not clone their symbolic implementation")
        .def("add", [](ocp_base &self, expr_handle ex) { self.add(std::move(ex)); },
             nb::arg("ex"), "Add one expression by shared handle")
        .def("dim", [](ocp_base &self, field_t field) { return self.dim(field); },
             nb::arg("field"), "Get the total active dimension assigned to a solver field")
        .def("wait_until_ready", &ocp_base::wait_until_ready, "Wait until all expressions in the OCP problem are ready")
        .def("is_active", &ocp_base::is_active, nb::arg("arg"), "Check if a given argument is active in the OCP problem")
        .def("print_summary", &ocp_base::print_summary, "Print a summary of the OCP problem");

    nb::class_<ocp, ocp_base>(
        m, "ocp", "Internal finalized interval problem consumed by one solver node.");

    nb::class_<node_view>(
        m, "endpoint",
        "Graph boundary view accepting state-only costs and constraints authored on x.")
        .def("add", [](node_view &self, expr_inarg_list &&exprs) { self.add(exprs); },
             nb::arg("exprs"), "Add state-only expressions to this boundary; graph composition performs endpoint lowering")
        .def("add", [](node_view &self, expr_handle ex) { self.add(std::move(ex)); },
             nb::arg("ex"), "Add one state-only expression to this boundary; inputs and dynamics are rejected");

    nb::class_<stage_ocp, ocp>(
        m, "stage_ocp",
        "Authored interval stage containing dynamics, path terms, and start/end boundary views.")
        .def_static("create", &stage_ocp::create,
                    "Create an empty authored interval stage; prefer the public ``moto.stage()`` helper")
        .def("copy", [](const stage_ocp &self,
                         const expr_inarg_list &disable,
                         const expr_inarg_list &enable) {
            return self.copy(active_status(disable, enable));
        }, nb::arg("disable") = nb::list{}, nb::arg("enable") = nb::list{},
             "Create an independent stage container while sharing immutable expression handles; optionally change active expressions")
        .def("disable", [](stage_ocp &self, const expr_inarg_list &exprs) {
            self.update_active_status(active_status(exprs, {}));
        }, nb::arg("exprs"), "Disable stage expressions")
        .def("disable", [](stage_ocp &self, expr_handle ex) {
            ocp_base::active_status_config config;
            config.deactivate_list.push_back(std::move(ex));
            self.update_active_status(config);
        }, nb::arg("ex"), "Disable a stage expression")
        .def("enable", [](stage_ocp &self, const expr_inarg_list &exprs) {
            self.update_active_status(active_status({}, exprs));
        }, nb::arg("exprs"), "Enable stage expressions")
        .def("enable", [](stage_ocp &self, expr_handle ex) {
            ocp_base::active_status_config config;
            config.activate_list.push_back(std::move(ex));
            self.update_active_status(config);
        }, nb::arg("ex"), "Enable a stage expression")
        .def("add", [](stage_ocp &self, expr_inarg_list &&exprs) { self.add(exprs); },
             nb::arg("exprs"), "Add interval dynamics, costs, or path constraints")
        .def("add", [](stage_ocp &self, expr_handle ex) { self.add(std::move(ex)); },
             nb::arg("ex"), "Add one interval dynamics, cost, or path constraint")
        .def_prop_ro("st", [](stage_ocp &self) { return self.st(); },
                     "Incoming state-only boundary; on a connected graph it shares the predecessor's terminal state")
        .def_prop_ro("ed", [](stage_ocp &self) { return self.ed(); },
                     "Outgoing state-only boundary represented by this interval's terminal state");

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
        .def_prop_ro("value", [](node_data &self) -> auto & { return self.sym_val(); }, nb::rv_policy::reference_internal)
        .def("data", [](node_data &self, const generic_func &f) -> auto & { return self.data(f); },
             nb::arg("function"), nb::rv_policy::reference_internal);

    nb::class_<func_approx_data>(m, "func_approx_data")
        .def("__getitem__", [](func_approx_data &self, py_var_inarg_wrapper s) { return self[(sym &)s]; })
        .def_prop_ro("v", [](func_approx_data &self) -> auto { return self.v_; }, "Value vector reference")
        .def(
            "jac",
            [](func_approx_data &self, py_var_inarg_wrapper in) -> auto { return self.jac((sym &)in); },
            "Get the jacobian reference for the input variable")
        .def("set_jac", [](func_approx_data &self, py_var_inarg_wrapper in, Eigen::Ref<const matrix> rhs) { self.jac((sym &)in) = rhs; });
}
