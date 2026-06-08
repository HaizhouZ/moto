
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>
#include <type_cast.hpp>

#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>

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
} // namespace moto

namespace {
moto::ineq_constr::box_bound_t cast_box_bound(const nb::handle &h) {
    using namespace moto;

    if (nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h)) {
        return nb::cast<scalar_t>(h);
    }
    if (nb::hasattr(h, "this")) {
        return nb::cast<cs::SX>(h);
    }
    try {
        return nb::cast<vector>(h);
    } catch (const nb::cast_error &) {
        auto values = nb::cast<std::vector<scalar_t>>(h);
        vector out(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            out(static_cast<Eigen::Index>(i)) = values[i];
        }
        return out;
    }
}
} // namespace
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

void register_submodule_functional(nb::module_ &m) {
    using namespace moto;
    export_enum<moto::approx_order>(m);

    nb::class_<expr>(m, "expr")
        .def("__bool__", &expr::operator bool)
        .def("__str__", [](const expr &self) {
            return fmt::format("expr({:p}, name={}, uid={}, dim={}, field={})",
                               static_cast<const void *>(&self), self.name(), self.uid(), self.dim(), self.field());
        })
        .def_prop_ro("name", &expr::__get_name)
        .def_prop_ro("field", &expr::__get_field)
        .def_prop_ro("dim", &expr::__get_dim)
        .def_prop_ro("uid", [](const expr &self) { return size_t(self.uid()); })
        .def("finalize", [](expr &self, bool block_until_ready) { return self.finalize(block_until_ready); }, nb::arg("block_until_ready") = true)
        .def_prop_ro("tdim", &expr::__get_tdim);

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

    nb::class_<generic_func, expr>(m, "func")
        .def_prop_ro("in_args", [](generic_func &self) -> auto & { return static_cast<const std::vector<var> &>(self.in_args()); }, nb::rv_policy::reference_internal)
        .def_rw("value", &generic_func::value)
        .def_rw("jacobian", &generic_func::jacobian)
        .def_rw("hessian", &generic_func::hessian)
        .def_prop_ro("order", &generic_func::__get_order)
        .def("__str__", [](const generic_func &f) { return fmt::format("func(name='{}', uid={}, order={}, dim={}, field={})",
                                                                       f.name(), f.uid(), f.order(), f.dim(), f.field()); })
        .def("enable_if_all", [](generic_func &self, const expr_inarg_list &args) { self.enable_if_all(args); }, nb::arg("args"))
        .def("disable_if_any", [](generic_func &self, const expr_inarg_list &args) { self.disable_if_any(args); }, nb::arg("args"))
        .def("enable_if_any", [](generic_func &self, const expr_inarg_list &args) { self.enable_if_any(args); }, nb::arg("args"))
        .def("add_argument", [](generic_func &self, py_var_inarg_wrapper v) { self.add_argument((sym &)v); }, nb::arg("in"))
        .def("add_arguments", [](generic_func &self, const var_inarg_list &args) { self.add_arguments(args); })
        .def("remap_arguments",
             [](generic_func &self,
                const std::vector<std::pair<py_var_inarg_wrapper, py_var_inarg_wrapper>> &remap)
                 -> std::shared_ptr<generic_func> {
                 generic_func::symbol_remap cpp_remap;
                 cpp_remap.reserve(remap.size());
                 for (const auto &[from, to] : remap) {
                     cpp_remap.emplace_back(var((sym &)from), var((sym &)to));
                 }
                 auto remapped = self.remap_arguments(cpp_remap).cast<generic_func>();
                 if (remapped->finalized() && !remapped->wait_until_ready()) {
                     throw std::runtime_error(fmt::format("remapped function {} is not ready", remapped->name()));
                 }
                 return remapped;
             },
             nb::arg("remap"));

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
        .def(
            "cast_soft",
            [](generic_constr &self, const std::string &type_name) {
                return std::shared_ptr<generic_constr>(self.cast_soft(type_name));
            },
            nb::arg("type_name") = "pmm_constr");

    nb::class_<ineq_constr, generic_constr>(m, "ineq")
        .def_static(
            "create",
            [](const std::string &name, const var_inarg_list &args, const cs::SX &out, approx_order order, field_t field) {
                return std::shared_ptr<generic_constr>(ineq_constr::create(name, args, out, order, field));
            },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def_static(
            "create",
            [](const std::string &name, approx_order order, size_t dim, field_t field) {
                return std::shared_ptr<generic_constr>(ineq_constr::create(name, order, dim, field));
            },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined)
        .def_static(
            "create",
            [](const std::string &name,
               const var_inarg_list &args,
               const cs::SX &out,
               const nb::handle &lb,
               const nb::handle &ub,
               approx_order order,
               field_t field) {
                return std::shared_ptr<generic_constr>(
                    ineq_constr::create(name, args, out, cast_box_bound(lb), cast_box_bound(ub), order, field));
            },
            nb::arg("name"), nb::arg("in_args"), nb::arg("out"), nb::arg("lb"), nb::arg("ub"),
            nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined);

    nb::class_<moto::pmm_constr, generic_constr>(m, "pmm_constr")
        .def_rw("rho", &moto::pmm_constr::rho, "Dual penalty weight for the proximal multiplier method");

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
        .def("set_diag_hess",
             [](generic_cost &self) { return self.set_diag_hess(); })
        .def("set_gauss_newton",
             [](generic_cost &self, const py_var_inarg_wrapper &v) { return self.set_gauss_newton(var((sym &)v)); });

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
        .def("mark_shared_inputs", &dense_dynamics::mark_shared_inputs, nb::arg("shared_inputs"));
}
