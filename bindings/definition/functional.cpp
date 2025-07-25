#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/usr_func.hpp>
#include <nanobind/stl/function.h>
#include <type_cast.hpp>

#include <definition/sym_type_caster.hpp>

moto::sym_in_list::operator moto::expr_list() const {
    expr_list tmp;
    tmp.reserve(this->size());
    for (auto &ex : *this) {
        auto type = ex.type();
        if (nb::isinstance<moto::shared_expr>(ex)) {
            tmp.push_back(nb::cast<moto::shared_expr &>(ex));
        } else if (nb::hasattr(ex, "sym_base")) {
            tmp.push_back(nb::cast<moto::shared_expr &>(ex.attr("sym_base")));
        } else if (type.is(nb::type<moto::expr>())) {
            tmp.push_back(nb::cast<moto::expr &>(ex));
        } else if (type.is(nb::type<moto::func>())) {
            tmp.push_back(nb::cast<moto::func &>(ex));
        } else if (type.is(nb::type<moto::cost>())) {
            tmp.push_back(nb::cast<moto::cost &>(ex));
        } else if (type.is(nb::type<moto::constr>())) {
            tmp.push_back(nb::cast<moto::constr &>(ex));
        } else if (type.is(nb::type<moto::pre_compute>())) {
            tmp.push_back(nb::cast<moto::pre_compute &>(ex));
        } else if (type.is(nb::type<moto::usr_func>())) {
            tmp.push_back(nb::cast<moto::usr_func &>(ex));
        } else if (type.is(nb::type<moto::custom_func>())) {
            tmp.push_back(nb::cast<moto::custom_func &>(ex));
        } else {
            nb::print("Unknown type in sym_in_list: ", ex);
        }
    }
    return tmp;
}

void export_order_with_magic(nb::module_ &m, const std::string &python_name) {
    nb::enum_<moto::approx_order> enum_binder(m, python_name.c_str());

    // Iterate over all enum values provided by magic_enum
    for (auto [value, name] : magic_enum::enum_entries<moto::approx_order>()) {
        enum_binder.value(("approx_order_" + std::string(name)).c_str(), value);
    }
    enum_binder.export_values(); // Makes enum members accessible like MyEnum.MEMBER
}

#define DEF_IMPL_ATTR_GETTER(name, cls) \
    def_prop_rw(#name, [](cls &self) { return self.name(); }, [](cls &self, const decltype(cls::impl::name##_) &value) { self.name() = value; })

#undef DEF_IMPL_GETTER
#define DEF_IMPL_GETTER(cls) \
    def("get_impl", [](cls &self) { return std::format("{:p}", static_cast<const void *>(&self.get_impl())); })

#define DEF_REF_AS(cls) \
    def("as_" #cls, [](const shared_expr &self) -> cls & { return static_cast<cls &>(self); })

void register_submodule_functional(nb::module_ &m) {
    using namespace moto;
    export_order_with_magic(m, "approx_order");
    nb::class_<shared_expr>(m, "shared_expr")
        .def("__bool__", &shared_expr::operator bool)
        .def("__str__", [](const shared_expr &self) {
            return fmt::format("shared_expr({:p}, use_count={}, impl_use_count={})",
                               static_cast<const void *>(&self),
                               self.use_count(), self ? self->impl_use_count() : 0);
        })
        .DEF_REF_AS(expr)
        .DEF_REF_AS(sym)
        .DEF_REF_AS(func)
        .DEF_REF_AS(custom_func)
        .DEF_REF_AS(cost)
        .DEF_REF_AS(constr)
        .DEF_REF_AS(pre_compute)
        .DEF_REF_AS(usr_func)
        .def_prop_ro("use_count", &shared_expr::use_count)
        .def_prop_ro("impl_use_count", [](const shared_expr &self) { return self ? self->impl_use_count() : 0; });

    nb::class_<expr>(m, "expr")
        .DEF_IMPL_ATTR_GETTER(name, expr)
        .DEF_IMPL_ATTR_GETTER(field, expr)
        .DEF_IMPL_ATTR_GETTER(dim, expr)
        .DEF_IMPL_ATTR_GETTER(uid, expr)
        .def("add_dep", &expr::add_dep<expr>, nb::arg("ex"))
        .def("finalize", &expr::finalize)
        .DEF_IMPL_ATTR_GETTER(finalized, expr)
        .def("__bool__", &expr::operator bool)
        .def("set_impl", &expr::set_impl)
        .def("use_count", &expr::use_count)
        .def("impl_use_count", &expr::impl_use_count)
        .DEF_IMPL_GETTER(expr);

    m.def("create_sym", [](const std::string &name, size_t dim, moto::field_t field) { nb::print("create called\n"); return shared_expr(moto::sym(name, dim, field)); }, nb::arg("name"), nb::arg("dim") = moto::dim_tbd, nb::arg("field") = moto::field_t::__undefined);
    m.def("get_sym_sx", [](shared_expr &&s) {
        auto& ex = static_cast<moto::sym &>(s); 
        // fmt::print("get_sym_sx called {}\n", s.get_impl().weak_from_this().use_count());
        return cs::SX(ex.get_impl()); }, nb::arg("s"));
    m.def("create_states", [](const std::string &name, size_t dim) {
        auto [x, y] = moto::sym::states(name, dim);
        return std::make_pair(shared_expr(std::move(x)), shared_expr(std::move(y))); }, nb::arg("name"), nb::arg("dim") = moto::dim_tbd);
    m.def("print_all_sx", [](sym_in_list &&s) {
        for (sym &ex : *s) {
            std::cout << ex.name() << ": " << ex << '\n';
        }
    });
    // nb::class_<sym, expr>(m, "sym_base")
    //     .def(nb::init<sym &&>(), nb::rv_policy::take_ownership)
    //     .def(nb::init<const std::string &, size_t, moto::field_t>(), nb::arg("name"), nb::arg("dim") = dim_tbd, nb::arg("field") = moto::field_t::__undefined)
    //     .def("__str__", [](const sym &s) { return s.name(); })
    //     .def("__repr__", [](const sym &s) { return fmt::format("sym(name='{}', field={}, dim={})", s.name(), field::name(s.field()), s.dim()); })
    //     .def_static("inputs", &sym::inputs, nb::arg("name"), nb::arg("dim") = 1)
    //     .def_static("params", &sym::params, nb::arg("name"), nb::arg("dim") = 1)
    //     .def_static("states", &sym::states, nb::arg("name"), nb::arg("dim") = 1)
    //     .def("next", &sym::next)
    //     .def("prev", &sym::prev)
    //     .def("sx", [](const sym &s) { return cs::SX(s); })
    //     .DEF_IMPL_GETTER(sym);

    nb::class_<func, expr>(m, "func")
        .def(nb::init<const std::string &, moto::approx_order, size_t, field_t>(),
             nb::arg("name"), nb::arg("order") = moto::approx_order::first, nb::arg("dim") = 0, nb::arg("field") = moto::field_t::__undefined)
        .def(nb::init<const std::string &, const sym_list &, const cs::SX &, moto::approx_order, field_t>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = moto::approx_order::first, nb::arg("field") = moto::field_t::__undefined)
        .def("__str__", [](const func &f) { return fmt::format("func(name='{}', order={}, dim={})", f.name(), magic_enum::enum_name<approx_order>(f.order()), f.dim()); })
        .def("add_argument", &func::add_argument<sym>, nb::arg("in"))
        .def("add_arguments", nb::overload_cast<const sym_list &>(&func::add_arguments), nb::arg("args"))
        .def("create_approx_map", &func::create_approx_map, nb::arg("primal"), nb::arg("raw"), nb::arg("shared"))
        .def("compute_approx", &func::compute_approx<func_approx_map &>, nb::arg("data"), nb::arg("eval_val") = true, nb::arg("eval_jac") = false, nb::arg("eval_hess") = false)
        .DEF_IMPL_ATTR_GETTER(value, func)
        .DEF_IMPL_ATTR_GETTER(jacobian, func)
        .DEF_IMPL_ATTR_GETTER(hessian, func)
        .DEF_IMPL_GETTER(func);

    nb::class_<custom_func, func>(m, "custom_func")
        .def(nb::init<const std::string &, moto::approx_order, size_t, field_t>(),
             nb::arg("name"), nb::arg("order") = moto::approx_order::first, nb::arg("dim") = 0, nb::arg("field") = moto::field_t::__undefined)
        .def(nb::init<const std::string &, const sym_list &, const cs::SX &, moto::approx_order, field_t>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = moto::approx_order::first, nb::arg("field") = moto::field_t::__undefined)
        .def("create_arg_map", &custom_func::create_arg_map, nb::arg("primal"), nb::arg("shared"))
        .DEF_IMPL_ATTR_GETTER(create_custom_data, custom_func)
        .DEF_IMPL_ATTR_GETTER(custom_call, custom_func)
        .DEF_IMPL_GETTER(custom_func);

    nb::class_<cost, func>(m, "cost")
        .def(nb::init<const std::string &, moto::approx_order>(), nb::arg("name"), nb::arg("order") = moto::approx_order::second)
        .def(nb::init<const std::string &, const sym_list &, const cs::SX &, moto::approx_order>(), nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = moto::approx_order::second)
        .def("as_terminal", &cost::as_terminal)
        .DEF_IMPL_GETTER(cost);

    nb::class_<constr, func>(m, "constr")
        .def(nb::init<const std::string &, approx_order, size_t, field_t>(),
             nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0, nb::arg("field") = field_t::__undefined)
        .def(nb::init<const std::string &, const sym_list &, const cs::SX &, approx_order, field_t>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
             nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def("as_eq", &constr::as_eq<constr>, nb::arg("soft") = false)
        .def("as_ineq", [](constr &self, const std::string &type_name) { return self.as_ineq(type_name); }, nb::arg("type_name") = "ipm")
        .DEF_IMPL_GETTER(constr);

    nb::class_<usr_func, custom_func>(m, "usr_func")
        .def(nb::init<const std::string &, approx_order, size_t>(),
             nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = 0)
        .def(nb::init<const std::string &, const sym_list &, const cs::SX &, approx_order>(),
             nb::arg("name"), nb::arg("in_args"), nb::arg("out"),
             nb::arg("order") = approx_order::first)
        .DEF_IMPL_GETTER(usr_func);

    nb::class_<pre_compute, custom_func>(m, "pre_compute")
        .def(nb::init<const std::string &>(), nb::arg("name"))
        .DEF_IMPL_GETTER(pre_compute);
}
