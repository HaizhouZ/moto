
#include <nanobind/stl/function.h>
#include <type_cast.hpp>

#include <definition/sym_type_caster.hpp>

moto::expr_in_list::operator moto::expr_list() const {
    moto::expr_list tmp;
    tmp.reserve(this->size());
    for (auto &ex : *this) {
        auto type = ex.type();
        if (nb::isinstance<moto::shared_expr>(ex)) {
            tmp.push_back(nb::cast<moto::shared_expr &>(ex));
        } else if (nb::hasattr(ex, "sym_base")) {
            tmp.push_back(nb::cast<moto::var &>(ex.attr("sym_base")));
            // } else if (type.is(nb::type<moto::expr>())) {
            //     tmp.push_back(nb::cast<moto::expr &>(ex));
            // } else if (type.is(nb::type<moto::func_base>())) {
            //     tmp.push_back(nb::cast<moto::func_base &>(ex));
            // } else if (type.is(nb::type<moto::cost>())) {
            //     tmp.push_back(nb::cast<moto::cost &>(ex));
            // } else if (type.is(nb::type<moto::constr>())) {
            //     tmp.push_back(nb::cast<moto::constr &>(ex));
            // } else if (type.is(nb::type<moto::pre_compute>())) {
            //     tmp.push_back(nb::cast<moto::pre_compute &>(ex));
            // } else if (type.is(nb::type<moto::usr_func>())) {
            //     tmp.push_back(nb::cast<moto::usr_func &>(ex));
            // } else if (type.is(nb::type<moto::custom_func>())) {
            //     tmp.push_back(nb::cast<moto::custom_func &>(ex));
        } else {
            nb::print("Unknown type in sym_in_list: ", ex);
        }
    }
    return tmp;
}

namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::var_list> {
    NB_TYPE_CASTER(moto::var_list, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) + const_name("[") +
                                       make_caster<moto::var>::Name + const_name(" | moto.sym") +
                                       const_name("]"))

    list_caster<std::vector<handle>, handle> list_cast;
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        if (!list_cast.from_python(src, flags, cleanup)) {
            return false;
        }
        auto &l = list_cast.value;
        value.clear();
        for (auto &ex : l) {
            auto type = ex.type();
            if (nb::isinstance<moto::shared_expr>(ex)) {
                value.push_back(nb::cast<moto::shared_expr &>(ex));
            } else if (nb::isinstance<moto::var>(ex)) {
                value.push_back(nb::cast<moto::var &>(ex));
            } else if (nb::hasattr(ex, "sym_base")) {
                value.push_back(nb::cast<moto::var &>(ex.attr("sym_base")));
            } else {
                nb::print("Unknown type in var_list caster: ", ex);
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
            return fmt::format("shared_expr({:p}, use_count={})",
                               static_cast<const void *>(&self), self.use_count());
        })
        .DEF_REF_AS(expr)
        .DEF_REF_AS(func_base)
        .DEF_REF_AS(custom_func)
        .DEF_REF_AS(cost)
        .DEF_REF_AS(constr)
        .DEF_REF_AS(pre_compute)
        .DEF_REF_AS(usr_func)
        .def_prop_ro("use_count", &shared_expr::use_count);

    nb::class_<var>(m, "var")
        .def("as_expr", [](const var &self) -> expr & { return static_cast<expr &>(self); }, nb::rv_policy::reference_internal)
        .def("__str__", [](const var &v) { return fmt::format("var(name='{}', dim={}, field={}, uid={})",
                                                              v->name(), v->dim(), field::name(v->field()), v->uid()); })
        .def_prop_ro("use_count", &var::use_count);

    nb::class_<expr>(m, "expr")
        .def_prop_rw("name", [](expr &self) { return self.name(); }, [](expr &self, const std::string &value) { self.name() = value; })
        .def_prop_rw("field", [](expr &self) { return self.field(); }, [](expr &self, const field_t &value) { self.field() = value; })
        .def_prop_rw("dim", [](expr &self) { return self.dim(); }, [](expr &self, const size_t &value) { self.dim() = value; })
        .def_prop_rw("uid", [](expr &self) { return self.uid(); }, [](expr &self, const size_t &value) { self.uid() = value; })
        .def("add_dep", nb::overload_cast<const shared_expr &>(&expr::add_dep<expr>), nb::arg("ex"))
        .def("finalize", &expr::finalize)
        .def_prop_rw("finalized", [](expr &self) { return self.finalized(); }, [](expr &self, const bool &value) { self.finalized() = value; })
        .def("__bool__", &expr::operator bool)
        .def("set_impl", &expr::set_impl)
        .def_prop_ro("use_count", &expr::use_count);

    m.def("create_sym", [](const std::string &name, size_t dim, field_t field) {
        nb::print("create called\n");
        return var(sym(name, dim, field)); }, nb::arg("name"), nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined);
    m.def("get_sym_sx", [](var &&s) {
        auto &ex = static_cast<sym &>(s);
        // fmt::print("get_sym_sx called {}\n", s.weak_from_this().use_count());
        return cs::SX(ex); }, nb::arg("s"));
    m.def("create_states", [](const std::string &name, size_t dim) {
        auto&& [x, y] = sym::states(name, dim);
        fmt::print("CPp: Created states: {}, {}\n", x->name(), y->name());
        return std::make_pair(var(std::move(x)), var(std::move(y))); }, nb::arg("name"), nb::arg("dim") = dim_tbd);
    m.def("print_all_sx", [](var_list &&s) {
        for (sym &ex : s) {
            std::cout << ex.name() << ": " << ex << '\n';
        }
    });

    using func_callback_t = std::function<void(func_approx_map &)>;

    nb::class_<func>(m, "func")
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
            "name",
            [](func &self) { return self->name(); },
            [](func &self, const std::string &value) { self->name() = value; })
        .def_prop_rw(
            "order",
            [](func &self) { return self->order(); },
            [](func &self, const approx_order &value) { self->order() = value; })
        .def_prop_rw(
            "dim",
            [](func &self) { return self->dim(); },
            [](func &self, const size_t &value) { self->dim() = value; })
        .def_prop_rw(
            "field",
            [](func &self) { return self->field(); },
            [](func &self, const field_t &value) { self->field() = value; })
        .def_prop_rw(
            "uid",
            [](func &self) { return self->uid(); },
            [](func &self, const size_t &value) { self->uid() = value; })
        .def_prop_rw(
            "custom_call",
            [](func &self) { return static_cast<custom_func &>(self).custom_call; },
            [](func &self, const decltype(custom_func::custom_call) &v) { static_cast<custom_func &>(self).custom_call = v; })
        .def("__str__",
             [](const func &f) { return fmt::format("func(name='{}', order={}, dim={}, field={})",
                                                    f->name(), magic_enum::enum_name<approx_order>(f->order()), f->dim(), field::name(f->field())); })
        .def(
            "as_eq",
            [](func &self, bool soft) { return func(static_cast<constr &>(self).as_eq<constr>(soft)); },
            nb::arg("soft") = false)
        .def(
            "as_ineq",
            [](constr &self, const std::string &type_name) { return func(static_cast<constr &>(self).as_ineq(type_name)); },
            nb::arg("type_name") = "ipm")
        .def_prop_rw(
            "create_custom_data",
            [](func &self) { return static_cast<custom_func &>(self).create_custom_data; },
            [](func &self, const decltype(custom_func::create_custom_data) &v) { static_cast<custom_func &>(self).create_custom_data = v; })
        .def(
            "add_argument",
            [](func &self, const var &v) { self->add_argument(v); },
            nb::arg("in"))
        .def(
            "add_arguments",
            [](func &self, const var_list &args) { self->add_arguments(args); })
        .def(
            "create_approx_map",
            [](func &self, sym_data &primal, dense_approx_data &raw, shared_data &shared) { return self->create_approx_map(primal, raw, shared); },
            nb::arg("primal"), nb::arg("raw"), nb::arg("shared"));

    m.def(
         "constr",
         [](const std::string &name, const var_list &in_args, const cs::SX &out,
            approx_order order = approx_order::first, field_t field = field_t::__undefined) { return func(moto::constr(name, in_args, out, order, field)); },
         nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def(
            "constr",
            [](const std::string &name, approx_order order, size_t dim, field_t field) { return func(moto::constr(name, order, dim, field)); },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined)
        .def(
            "cost",
            [](const std::string &name, const var_list &in_args, const cs::SX &out, approx_order order = approx_order::second) { return func(moto::cost(name, in_args, out, order)); },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::second)
        .def(
            "cost",
            [](const std::string &name, approx_order order = approx_order::second) { return func(moto::cost(name, order)); },
            nb::arg("name"), nb::arg("order") = approx_order::second)
        .def(
            "usr_func",
            [](const std::string &name, const var_list &in_args, const cs::SX &out, approx_order order = approx_order::first) { return func(moto::usr_func(name, in_args, out, order)); },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first)
        .def(
            "usr_func",
            [](const std::string &name, approx_order order = approx_order::first) { return func(moto::usr_func(name, order)); },
            nb::arg("name"), nb::arg("order") = approx_order::first)
        .def(
            "pre_compute",
            [](const std::string &name) { return func(moto::pre_compute(name)); }, nb::arg("name"));

    // .def(nb::init<const std::string &, approx_order, size_t, field_t>(),
    //      nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0, nb::arg("field") = field_t::__undefined)
    // .def(nb::init<const std::string &, const var_list &, const cs::SX &, approx_order, field_t>(),
    //      nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
    // .def("__str__", [](const func_base &f) {
    //     return fmt::format("func_base(name='{}', order={}, dim={})", f.name(), magic_enum::enum_name<approx_order>(f.order()), f.dim());
    // })
    // .def("add_argument", &func_base::add_argument<var>, nb::arg("in"))
    // .def("add_arguments", nb::overload_cast<const var_list &>(&func_base::add_arguments), nb::arg("args"))
    // .def("create_approx_map", &func_base::create_approx_map, nb::arg("primal"), nb::arg("raw"), nb::arg("shared"))
    // .def("compute_approx", &func_base::compute_approx<func_approx_map &>, nb::arg("data"), nb::arg("eval_val") = true, nb::arg("eval_jac") = false, nb::arg("eval_hess") = false)
    // .def_prop_rw("value", [](func_base &self) { return self.value; }, [](func_base &self, const func_callback_t &v) { self.value = v; })
    // .def_prop_rw("jacobian", [](func_base &self) { return self.jacobian; }, [](func_base &self, const func_callback_t &v) { self.jacobian = v; })
    // .def_prop_rw("hessian", [](func_base &self) { return self.hessian; }, [](func_base &self, const func_callback_t &v) { self.hessian = v; });

    // nb::class_<custom_func, func_base>(m, "custom_func")
    //     .def(nb::init<const std::string &, approx_order, size_t, field_t>(),
    //          nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0, nb::arg("field") = field_t::__undefined)
    //     .def(nb::init<const std::string &, const var_list &, const cs::SX &, approx_order, field_t>(),
    //          nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
    //     .def("create_arg_map", &custom_func::create_arg_map, nb::arg("primal"), nb::arg("shared"))
    //     .def_prop_rw(
    //         "create_custom_data",
    //         [](custom_func &self) { return self.create_custom_data; },
    //         [](custom_func &self, const decltype(custom_func::create_custom_data) &v) { self.create_custom_data = v; })
    //     .def_prop_rw(
    //         "custom_call",
    //         [](custom_func &self) { return self.custom_call; },
    //         [](custom_func &self, const decltype(custom_func::custom_call) &v) { self.custom_call = v; });

    // nb::class_<cost, func_base>(m, "cost")
    //     .def(nb::init<const std::string &, approx_order>(), nb::arg("name"), nb::arg("order") = approx_order::second)
    //     .def(nb::init<const std::string &, const var_list &, const cs::SX &, approx_order>(), nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::second)
    //     .def("as_terminal", &cost::as_terminal);

    // nb::class_<constr, func_base>(m, "constr")
    //     .def(nb::init<const std::string &, approx_order, size_t, field_t>(),
    //          nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0, nb::arg("field") = field_t::__undefined)
    //     .def(nb::init<const std::string &, const var_list &, const cs::SX &, approx_order, field_t>(),
    //          nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
    //          nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
    //     .def("as_eq", &constr::as_eq<constr>, nb::arg("soft") = false)
    //     .def("as_ineq", [](constr &self, const std::string &type_name) { return self.as_ineq(type_name); }, nb::arg("type_name") = "ipm");
    //         nb::object a;

    // nb::class_<usr_func, custom_func>(m, "usr_func")
    //     .def(nb::init<const std::string &, approx_order, size_t>(),
    //          nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0)
    //     .def(nb::init<const std::string &, const var_list &, const cs::SX &, approx_order>(),
    //          nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
    //          nb::arg("order") = approx_order::first);

    // nb::class_<pre_compute, custom_func>(m, "pre_compute")
    //     .def(nb::init<const std::string &>(), nb::arg("name"));

    // struct func_base : public shared_object<func_base> {
    //     using shared_object<func_base>::shared_object;
    // };
    // var v = sym::inputs("v", 1);
    // func_base f = func_base("test_func", {v}, v * 2, approx_order::first);

    // sym s = sym("s", 1, field_t::__u);
    // var sv(s);
}
