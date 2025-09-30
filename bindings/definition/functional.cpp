
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/ocp/usr_func.hpp>
#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>
#include <type_cast.hpp>

#include <moto/ocp/dynamics/dense_dynamics.hpp>

namespace moto {
shared_expr &cast_to_shared_expr(const nb::handle &h) {
    if (nb::isinstance<moto::shared_expr>(h)) {
        return nb::cast<moto::shared_expr &>(h);
    } else if (nb::hasattr(h, "sym_base")) {
        return nb::cast<moto::var &>(h.attr("sym_base"));
    } else {
        nb::print("Unsupported type for cast_to_var: ", h);
        throw std::runtime_error("Unsupported type for cast_to_shared_expr");
    }
}
var &cast_to_var(const nb::handle &h) {
    if (nb::isinstance<moto::var>(h)) {
        return nb::cast<moto::var &>(h);
    } else if (nb::hasattr(h, "sym_base")) {
        return nb::cast<moto::var &>(h.attr("sym_base"));
    } else {
        nb::print("Unsupported type for cast_to_var: ", h);
        throw std::runtime_error("Unsupported type for cast_to_var");
    }
}
func &cast_to_func(const nb::handle &h) {
    if (nb::isinstance<moto::func>(h)) {
        return nb::cast<moto::func &>(h);
    } else if (nb::hasattr(h, "sym_base")) {
        return nb::cast<moto::func &>(h.attr("sym_base"));
    } else {
        nb::print("Unsupported type for cast_to_func: ", h);
        throw std::runtime_error("Unsupported type for cast_to_func");
    }
}
} // namespace moto
namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::var_inarg_list> {
    NB_TYPE_CASTER(moto::var_inarg_list, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) + const_name("[") +
                                             make_caster<moto::var>::Name + const_name(" | casadi.SX") +
                                             const_name("]"))

    list_caster<std::vector<handle>, handle> list_cast;
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        if (!list_cast.from_python(src, flags, cleanup)) {
            return false;
        }
        auto &l = list_cast.value;
        value.reserve(l.size());
        value.clear();
        for (auto &ex : l) {
            try {
                value.push_back(moto::cast_to_var(ex));
            } catch (const std::exception &e) {
                fmt::print("Failed to cast to var: {}\n", e.what());
                return false;
            }
        }
        return true;
    }
};

} // namespace detail
} // namespace nanobind

void export_order_with_magic(nb::module_ &m, const std::string &python_name) {
    nb::enum_<moto::approx_order> enum_binder(m, python_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<moto::approx_order>()) {
        enum_binder.value(("approx_order_" + std::string(name)).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

#undef DEF_IMPL_GETTER
#define DEF_IMPL_GETTER(cls)                                                     \
    def("get_impl", [](cls &self) {                                              \
        return std::format("{:p}", static_cast<const void *>(&self.get_impl())); \
    })

#define DEF_REF_AS(cls) \
    def("as_" #cls, [](const shared_expr &self) -> cls & { return static_cast<cls &>(self); }, nb::rv_policy::reference_internal)

#define DEF_REF_ATTR(name, cls) \
    def_prop_rw(name, [](cls &self) { return self->name(); }, [](cls &self, const std::string &value) { self->name() = value; })

void register_submodule_functional(nb::module_ &m) {
    using namespace moto;
    export_order_with_magic(m, "approx_order");
    nb::class_<shared_expr>(m, "shared_expr")
        .def("__bool__", &shared_expr::operator bool)
        .def("__str__", [](const shared_expr &self) {
            return fmt::format("shared_expr({:p}, name={}, uid={}, dim={}, field={}, use_count={})",
                               static_cast<const void *>(&self), self->name(), self->uid(), self->dim(), self->field(), self.use_count());
        })
        .def_prop_ro("use_count", &shared_expr::use_count)
        .def_prop_rw("name", [](shared_expr &self) { return self->name(); }, [](shared_expr &self, const std::string &value) { self->name() = value; })
        .def_prop_rw("field", [](shared_expr &self) { return self->field(); }, [](shared_expr &self, const field_t &value) { self->field() = value; })
        .def_prop_rw("dim", [](shared_expr &self) { return self->dim(); }, [](shared_expr &self, const size_t &value) { self->dim() = value; })
        .def_prop_ro("uid", [](shared_expr &self) { return self->uid(); })
        .def("finalize", [](shared_expr &self, bool block_until_ready) { self->finalize(block_until_ready); }, nb::arg("block_until_ready") = true)
        .def("wait_until_ready", [](shared_expr &self) { return self->wait_until_ready(); })
        .def_prop_ro("finalized", [](expr &self) { return self.finalized(); })
        .def_prop_rw("default_active_status", [](expr &self) { return self.default_active_status(); }, [](expr &self, bool value) { self.default_active_status() = value; })
        .def("__bool__", &shared_expr::operator bool);

    nb::class_<var, shared_expr>(m, "var")
        .def("__str__", [](const var &v) { return fmt::format("var(name='{}', dim={}, field={}, uid={})",
                                                              v->name(), v->dim(), v->field(), v->uid()); })
        .def_prop_rw("default_value", [](var &self) { return self->default_value(); }, [](var &self, const sym::default_val_t &value) { self->set_default_value(value); })
        .def("to_sx", [](var &v) { return (cs::SX &)static_cast<sym &>(v); }, nb::rv_policy::reference_internal)
        .def("clone", [](const var &self) { return self->clone(); })
        .def("symbolic_integrate", [](const var &self, const cs::SX &x, const cs::SX &dx) { return self->symbolic_integrate(x, dx); }, nb::arg("x"), nb::arg("dx"))
        .def("symbolic_difference", [](const var &self, const cs::SX &x1, const cs::SX &x0) { return self->symbolic_difference(x1, x0); }, nb::arg("x1"), nb::arg("x0"), "difference from x0 to x1, i.e., x1 - x0")
        .def("integrate", [](const var &self, moto::vector_ref x, moto::vector_ref dx, moto::scalar_t alpha) { 
            vector tmp(self->dim());
            self->integrate(x, dx, tmp, alpha);
            return tmp; }, nb::arg("x"), nb::arg("dx"), nb::arg("alpha") = 1.0)
        .def("difference", [](const var &self, moto::vector_ref x1, moto::vector_ref x0) { 
            vector tmp(self->dim());
            self->difference(x1, x0, tmp);
            return tmp; }, nb::arg("x1"), nb::arg("x0"));

    m.def(
        "create_sym",
        [](const std::string &name, size_t dim, field_t field, sym::default_val_t default_val) { return var(sym::symbol(name, dim, field, default_val)); },
        nb::arg("name"), nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined, nb::arg("default_val") = nb::none());
    m.def("get_sym_sx", [](var &&s) {
        auto &ex = static_cast<sym &>(s);
        return cs::SX(ex); }, nb::arg("s"));
    m.def(
        "create_states", [](const std::string &name, size_t dim, sym::default_val_t default_val = sym::default_val_none_t()) {
        auto&& [x, y] = sym::states(name, dim, default_val);
        return std::make_pair(var(std::move(x)), var(std::move(y))); },
        nb::arg("name"), nb::arg("dim") = dim_tbd, nb::arg("default_val") = nb::none());
    m.def("print_all_sx", [](var_inarg_list &&s) {
        for (sym &ex : s) {
            std::cout << ex.name() << ": " << ex << '\n';
        }
    });

    using func_callback_t = std::function<void(func_approx_data &)>;

    nb::class_<func, shared_expr>(m, "func")
        .def_prop_rw(
            "value",
            [](func &self) { return self->value; },
            [](func &self, const func_callback_t &v) { self->value = v; })
        .def_prop_rw(
            "jacobian",
            [](func &self) { return self->jacobian; },
            [](func &self, const func_callback_t &v) { self->jacobian = v; })
        .def_prop_rw(
            "hessian",
            [](func &self) { return self->hessian; },
            [](func &self, const func_callback_t &v) { self->hessian = v; })
        .def_prop_rw(
            "order",
            [](func &self) { return self->order(); },
            [](func &self, const approx_order &value) { self->order() = value; })
        .def("__str__",
             [](const func &f) { return fmt::format("func(name='{}', uid={}, order={}, dim={}, field={})",
                                                    f->name(), f->uid(), f->order(), f->dim(), f->field()); })
        .def("clone", [](const func &self) { return self->clone(); })
        .def("finalize", [](func &self, bool block_until_ready) { self->finalize(block_until_ready); }, nb::arg("block_until_ready") = true)
        .def("enable_if_all", [](func &self, const expr_inarg_list &args) { self->enable_if_all(args); }, nb::arg("args"))
        .def("disable_if_any", [](func &self, const expr_inarg_list &args) { self->disable_if_any(args); }, nb::arg("args"))
        .def("enable_if_any", [](func &self, const expr_inarg_list &args) { self->enable_if_any(args); }, nb::arg("args"))
        .def("add_argument", [](func &self, const py_var_wrapper &v) { self->add_argument((var &)v); }, nb::arg("in"))
        .def("add_arguments", [](func &self, const var_inarg_list &args) { self->add_arguments(args); })
        .def("create_approx_data", [](func &self, sym_data &primal, merit_data &raw, shared_data &shared) { return self->create_approx_data(primal, raw, shared); }, nb::arg("primal"), nb::arg("raw"), nb::arg("shared"));

    nb::class_<constr, func>(m, "constr")
        .def(
            nb::init<const std::string &, const var_inarg_list &, const cs::SX &, approx_order, field_t>(),
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def(
            nb::init<const std::string &, approx_order, size_t, field_t>(),
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined)
        .def("clone", [](const constr &self) { return self->clone(); })
        .def(
            "as_eq",
            [](constr &self, bool soft) { return self.as_soft(); },
            nb::arg("soft") = false, nb::rv_policy::move)
        .def(
            "as_ineq",
            [](constr &self, const std::string &type_name) { return self.as_ineq(type_name); },
            nb::arg("type_name") = "ipm", nb::rv_policy::move);

    nb::class_<cost, func>(m, "cost")
        .def(
            nb::init<const std::string &, const var_inarg_list &, const cs::SX &, approx_order>(),
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::second)
        .def(
            nb::init<const std::string &, approx_order>(),
            nb::arg("name"), nb::arg("order") = approx_order::second)
        .def("set_diag_hess", &cost::set_diag_hess, nb::rv_policy::move)
        .def(
            "as_terminal",
            [](cost &self) { return self.as_terminal(); }, nb::rv_policy::move)
        .def("set_gauss_newton", [](cost &self) { return self.set_gauss_newton(); }, nb::rv_policy::move)
        .def("clone", [](const cost &self) { return self->clone(); });

    // nb::class_<custom_func, func>(m, "custom_func")
    //     .def_prop_rw(
    //         "custom_call",
    //         [](custom_func &self) { return self->custom_call; },
    //         [](custom_func &self, const decltype(generic_custom_func::custom_call) &v) { self->custom_call = v; })
    //     .def_prop_rw(
    //         "create_custom_data",
    //         [](custom_func &self) { return self->create_custom_data; },
    //         [](custom_func &self, const decltype(generic_custom_func::create_custom_data) &v) { self->create_custom_data = v; });

    // nb::class_<usr_func, custom_func>(m, "usr_func")
    //     .def(
    //         nb::init<const std::string &, const var_inarg_list &, const cs::SX &, approx_order>(),
    //         nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first)
    //     .def(
    //         nb::init<const std::string &, approx_order, size_t>(),
    //         nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd);

    // nb::class_<pre_compute, custom_func>(m, "pre_compute")
    //     .def(
    //         nb::init<const std::string &>(),
    //         nb::arg("name") = "pre_compute");

    nb::class_<dense_dynamics, func>(m, "dense_dynamics")
        .def(
            nb::init<const std::string &, const var_inarg_list &, const cs::SX &, approx_order>(),
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first)
        .def(
            nb::init<const std::string &, approx_order, size_t>(),
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd);
}
