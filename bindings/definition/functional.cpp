
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/ocp/usr_func.hpp>
#include <moto/solver/soft_constr/quadratic_penalized.hpp>
#include <type_cast.hpp>

#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>

#include <moto/ocp/problem.hpp>

#include <moto/ocp/dynamics/dense_dynamics.hpp>

#include <enum_export.hpp>

namespace moto {
expr *get_expr_ptr(const nb::handle &h) {
    if (nb::isinstance<moto::expr>(h)) {
        return &nb::cast<moto::expr &>(h);
    } else if (nb::hasattr(h, "__sym__")) {
        return &static_cast<expr &>(nb::cast<moto::sym &>(h.attr("__sym__")));
    } else {
        nb::print("Unsupported type for cast_to_var: ", h);
        throw std::runtime_error("Unsupported type for cast_to_shared_expr");
    }
}
func &cast_to_func(const nb::handle &h) {
    if (nb::isinstance<moto::func>(h)) {
        return nb::cast<moto::func &>(h);
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
    NB_TYPE_CASTER(moto::var_inarg_list, io_name("collections.abc.Sequence", "list") + const_name("[") +
                                             make_caster<moto::var>::Name +
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
                moto::expr *ptr = moto::get_expr_ptr(ex);
                value.emplace_back(static_cast<moto::sym &>(*ptr));
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


#define TO_SHARED_PTR(cls, ptr) std::shared_ptr<cls>(static_cast<cls *>(ptr))

#define DEF_CLONE_FUNC(cls) \
    def("clone", [](const cls &self) { return TO_SHARED_PTR(cls, self.clone()); })

void register_submodule_functional(nb::module_ &m) {
    using namespace moto;
    export_enum<moto::approx_order>(m);

    nb::class_<expr>(m, "expr")
        .def("__bool__", &expr::operator bool)
        .def("__str__", [](const expr &self) {
            return fmt::format("expr({:p}, name={}, uid={}, dim={}, field={})",
                               static_cast<const void *>(&self), self.name(), self.uid(), self.dim(), self.field());
        })
        .def_prop_rw("name", &expr::__get_name, &expr::__set_name)
        .def_prop_rw("field", &expr::__get_field, &expr::__set_field)
        .def_prop_rw("dim", &expr::__get_dim, &expr::__set_dim)
        .def_prop_ro("uid", [](const expr &self) { return size_t(self.uid()); })
        .def("finalize", [](expr &self, bool block_until_ready) { return self.finalize(block_until_ready); }, nb::arg("block_until_ready") = true)
        .def("wait_until_ready", [](expr &self) { return self.wait_until_ready(); })
        .def_prop_ro("finalized", &expr::__get_finalized)
        .def_prop_rw("tdim", &expr::__get_tdim, &expr::__set_tdim)
        .def_prop_rw("default_active_status", &expr::__get_default_active_status, &expr::__set_default_active_status);

    nb::class_<sym, expr>(m, "sym")
        .def("__str__", [](const sym &v) { return fmt::format("sym(name='{}', dim={}, field={}, uid={})",
                                                              v.name(), v.dim(), v.field(), v.uid()); })
        .def_prop_rw("default_value", &sym::__get_default_value, &sym::__set_default_value)
        .def_prop_ro("sx", [](sym &v) { return (cs::SX &)v; }, nb::rv_policy::reference_internal)
        .def("clone", [](const sym &self, const std::string &name) { return self.clone(name); })
        .def("symbolic_integrate", [](const sym &self, const cs::SX &x, const cs::SX &dx) { return self.symbolic_integrate(x, dx); }, nb::arg("x"), nb::arg("dx"))
        .def("symbolic_difference", [](const sym &self, const cs::SX &x1, const cs::SX &x0) { return self.symbolic_difference(x1, x0); }, nb::arg("x1"), nb::arg("x0"), "difference from x0 to x1, i.e., x1 - x0")
        .def("integrate", [](const sym &self, moto::vector_ref x, moto::vector_ref dx, moto::scalar_t alpha) { 
            vector tmp(self.dim());
            self.integrate(x, dx, tmp, alpha);
            return tmp; }, nb::arg("x"), nb::arg("dx"), nb::arg("alpha") = 1.0)
        .def("difference", [](const sym &self, moto::vector_ref x1, moto::vector_ref x0) { 
            vector tmp(self.tdim());
            self.difference(x1, x0, tmp);
            return tmp; }, nb::arg("x1"), nb::arg("x0"))
        .def_static("symbol", &sym::symbol, nb::arg("name"), nb::arg("dim") = 1, nb::arg("field") = field_t::__undefined, nb::arg("default_val") = nb::none())
        .def_static("states", &sym::states, nb::arg("name"), nb::arg("dim") = 1, nb::arg("default_val") = nb::none())
        .def_static("inputs", &sym::inputs, nb::arg("name"), nb::arg("dim") = 1, nb::arg("default_val") = nb::none())
        .def_static("params", &sym::params, nb::arg("name"), nb::arg("dim") = 1, nb::arg("default_val") = nb::none());

    using func_callback_t = std::function<void(func_approx_data &)>;

    nb::class_<generic_func, expr>(m, "func")
        .def_prop_ro("in_args", [](generic_func &self) -> auto & { return static_cast<const std::vector<var> &>(self.in_args()); }, nb::rv_policy::reference_internal)
        .def_prop_ro("num_args", [](generic_func &self, field_t f) { return self.arg_num(f); })
        .def_rw("value", &generic_func::value)
        .def_rw("jacobian", &generic_func::jacobian)
        .def_rw("hessian", &generic_func::hessian)
        .def_prop_rw("order", &generic_func::__get_order, &generic_func::__set_order)
        .def("__str__", [](const generic_func &f) { return fmt::format("func(name='{}', uid={}, order={}, dim={}, field={})",
                                                                       f.name(), f.uid(), f.order(), f.dim(), f.field()); })
        .def("enable_if_all", [](generic_func &self, const expr_inarg_list &args) { self.enable_if_all(args); }, nb::arg("args"))
        .def("disable_if_any", [](generic_func &self, const expr_inarg_list &args) { self.disable_if_any(args); }, nb::arg("args"))
        .def("enable_if_any", [](generic_func &self, const expr_inarg_list &args) { self.enable_if_any(args); }, nb::arg("args"))
        .def("add_argument", [](generic_func &self, py_var_inarg_wrapper v) { self.add_argument((sym &)v); }, nb::arg("in"))
        .def("add_arguments", [](generic_func &self, const var_inarg_list &args) { self.add_arguments(args); })
        .def("active_dim", &generic_func::active_dim)
        .def("active_num", &generic_func::active_num)
        .def("active_tdim", &generic_func::active_tdim)
        .def("active_args", &generic_func::active_args)
        .DEF_CLONE_FUNC(generic_func)
        .def("create_approx_data", [](generic_func &self, sym_data &primal, merit_data &raw, shared_data &shared) { return self.create_approx_data(primal, raw, shared); }, nb::arg("primal"), nb::arg("raw"), nb::arg("shared"));

    nb::class_<generic_constr, generic_func>(m, "constr")
        .def_static(
            "create",
            [](const std::string &name, const var_inarg_list &args, const cs::SX &out, approx_order order, field_t field) {
                return std::make_shared<generic_constr>(name, args, out, order, field);
            },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def_static(
            "create",
            [](const std::string &name, approx_order order, size_t dim, field_t field) {
                return std::make_shared<generic_constr>(name, order, dim, field);
            },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined)
        .DEF_CLONE_FUNC(generic_constr)
        .def(
            "cast_ineq",
            [](generic_constr &self, const std::string &type_name) { return TO_SHARED_PTR(generic_constr, self.cast_ineq(type_name)); },
            nb::arg("type_name") = "ipm")
        .def(
            "cast_soft",
            [](generic_constr &self, const std::string &type_name) { return TO_SHARED_PTR(generic_constr, self.cast_soft(type_name)); },
            nb::arg("type_name") = "pmm_constr");

    nb::class_<moto::pmm_constr, generic_constr>(m, "pmm_constr")
        .def_rw("rho", &moto::pmm_constr::rho, "Dual penalty weight for the proximal multiplier method")
        .DEF_CLONE_FUNC(moto::pmm_constr);

    nb::class_<generic_cost, generic_func>(m, "cost")
        .def_static(
            "create",
            [](const std::string &name, const var_inarg_list &args, const cs::SX &out, approx_order order) {
                return std::make_shared<generic_cost>(name, args, out, order);
            },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::second)
        .def_static(
            "create",
            [](const std::string &name, approx_order order) {
                return std::make_shared<generic_cost>(name, order);
            },
            nb::arg("name"), nb::arg("order") = approx_order::second)
        .DEF_CLONE_FUNC(generic_cost)
        .def("set_diag_hess",
             [](generic_cost &self) { return self.set_diag_hess(); })
        .def("as_terminal",
             [](generic_cost &self) { return self.as_terminal(); })
        .def("set_gauss_newton",
             [](generic_cost &self, const py_var_inarg_wrapper &v) { return self.set_gauss_newton(var((sym &)v)); });

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

    nb::class_<dense_dynamics, generic_constr>(m, "dense_dynamics")
        .def_static(
            "create",
            [](const std::string &name, const var_inarg_list &args, const cs::SX &out, approx_order order) {
                return std::make_shared<dense_dynamics>(name, args, out, order);
            },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first)
        .def_static(
            "create",
            [](const std::string &name, approx_order order, size_t dim) {
                return std::make_shared<dense_dynamics>(name, order, dim);
            },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd)
        .def("active_dim_exclusive_inputs", &dense_dynamics::active_dim_exclusive_inputs, nb::arg("prob"))
        .def("active_dim_shared_inputs", &dense_dynamics::active_dim_shared_inputs, nb::arg("prob"))
        .def("active_num_exclusive_inputs", &dense_dynamics::active_num_exclusive_inputs, nb::arg("prob"))
        .def("active_num_shared_inputs", &dense_dynamics::active_num_shared_inputs, nb::arg("prob"))
        .def("mark_shared_inputs", &dense_dynamics::mark_shared_inputs, nb::arg("shared_inputs"));
}
