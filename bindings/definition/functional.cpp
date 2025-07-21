#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/usr_func.hpp>
// #include <nanobind/stl/

#include <binding_fwd.hpp>

#define access_prop_ro(name, prop) \
    def_prop_ro(name, [](Handle &self) { return self->prop; })
#define access_prop_rw(name, prop) \
    def_prop_rw(name, [](Handle &self) -> const auto & { return self->prop; }, [](Handle &self, const decltype(Handle::expr_type::prop) &value) { self->prop = value; })
#define access_method(name, prop) \
    def(name, [](Handle &self) { return self->prop(); })

template <typename Handle>
    requires std::derived_from<typename Handle::expr_type, moto::impl::expr>
auto export_expr_shared_wrapper(nb::module_ &m, const std::string &name) {
    using namespace moto;
    using namespace moto::impl;
    return nb::class_<Handle>(m, name.c_str())
        .access_prop_ro("name", name_)
        .access_prop_ro("field", field_)
        .access_prop_ro("dim", dim_)
        .access_prop_ro("uid", uid_);
}

template <typename Handle>
    requires std::derived_from<typename Handle::expr_type, moto::impl::func>
auto export_func_shared_wrapper(nb::module_ &m, const std::string &name) {
    using namespace moto;
    using namespace moto::impl;
    auto evaluate = [](Handle &self, moto::func_approx_map &data, bool eval_val, bool eval_jac, bool eval_hess) { self->compute_approx(data, eval_val, eval_jac, eval_hess); };
    auto create_approx_map = [](Handle &self, sym_data &primal, dense_approx_data &raw, shared_data &shared) { return self->create_approx_map(primal, raw, shared); };

    return export_expr_shared_wrapper<Handle>(m, name)
        .access_method("order", order)
        .access_prop_rw("value", value)
        .access_prop_rw("jacobian", jacobian)
        .access_prop_rw("hessian", hessian)
        .access_method("load_external", load_external)
        .def("add_arguments", [](Handle &self, const std::vector<moto::sym> &args) { self->add_arguments(args); })
        .def("create_approx_map", create_approx_map, nb::arg("primal"), nb::arg("raw"), nb::arg("shared"))
        .def("compute_approx", evaluate, nb::arg("data"), nb::arg("eval_val") = true, nb::arg("eval_jac") = false, nb::arg("eval_hess") = false)
        .def("get", [](Handle &self) -> Handle::expr_type & { return *self; });
}
template <typename Handle>
    requires std::derived_from<typename Handle::expr_type, moto::impl::custom_func>
auto export_custom_func_shared_wrapper(nb::module_ &m, const std::string &name) {
    using namespace moto;
    using namespace moto::impl;
    auto create_arg_map = [](Handle &self, sym_data &primal, shared_data &shared) {
        return self->create_arg_map(primal, shared);
    };
    return export_func_shared_wrapper<Handle>(m, name)
        .def("create_arg_map", create_arg_map, nb::arg("primal"), nb::arg("shared"))
        .access_prop_rw("create_custom_data", create_custom_data)
        .access_prop_rw("custom_call", custom_call);
}

void export_order_with_magic(nb::module_ &m, const std::string &python_name) {
    nb::enum_<moto::approx_order> enum_binder(m, python_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<moto::approx_order>()) {
        enum_binder.value(("approx_order_" + std::string(name)).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

void register_submodule_functional(nb::module_ &m) {
    using namespace moto;
    export_order_with_magic(m, "approx_order");
    nb::class_<impl::expr>(m, "impl_expr")
        .def_ro("name", &impl::expr::name_)
        .def_ro("field", &impl::expr::field_)
        .def_ro("dim", &impl::expr::dim_)
        .def_ro("uid", &impl::expr::uid_);

    export_expr_shared_wrapper<sym>(m, "sym")
        .def(nb::init<const std::string &, size_t, moto::field_t>(), nb::arg("name"), nb::arg("dim") = dim_tbd, nb::arg("field") = moto::field_t::__undefined)
        .def("__str__", [](const sym &s) { return s.name(); })
        .def("__repr__", [](const sym &s) { return fmt::format("sym(name='{}', field={}, dim={})", s.name(), field::name(s->field()), s->dim()); })
        .def_static("inputs", &sym::inputs, nb::arg("name"), nb::arg("dim") = 1)
        .def_static("params", &sym::params, nb::arg("name"), nb::arg("dim") = 1)
        .def_static("states", &sym::states, nb::arg("name"), nb::arg("dim") = 1);

    nb::class_<impl::func, impl::expr>(m, "impl_func");

    nb::class_<impl::custom_func, impl::func>(m, "impl_custom_func");

    export_func_shared_wrapper<constr>(m, "constr")
        .def(nb::init<const std::string &, approx_order, size_t, field_t>(),
             nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0, nb::arg("field") = field_t::__undefined)
        .def(nb::init<const std::string &, std::initializer_list<sym>, const cs::SX &, approx_order, field_t>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
             nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def("as_eq", &constr::as_eq<>, nb::arg("soft") = false)
        .def("as_ineq", nb::overload_cast<std::string_view>(&constr::as_ineq), nb::arg("type_name") = "moto::impl::ineq_constr");

    export_custom_func_shared_wrapper<usr_func>(m, "usr_func")
        .def(nb::init<const std::string &, approx_order, size_t>(),
             nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0)
        .def(nb::init<const std::string &, std::initializer_list<sym>, const cs::SX &, approx_order>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
             nb::arg("order") = approx_order::first);

    export_custom_func_shared_wrapper<pre_compute>(m, "pre_compute")
        .def(nb::init<const std::string &>(), nb::arg("name"));
}
