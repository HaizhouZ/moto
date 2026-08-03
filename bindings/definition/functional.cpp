
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/solver/soft_constr/pmm_constr.hpp>
#include <type_cast.hpp>

#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>

#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/semi_implicit_euler.hpp>

#include <enum_export.hpp>

namespace moto {
expr_handle get_expr_handle(const nb::handle &h) {
    if (nb::isinstance<moto::expr>(h)) {
        return nb::cast<moto::expr &>(h).handle();
    } else if (nb::hasattr(h, "__sym__")) {
        return nb::cast<moto::sym &>(h.attr("__sym__")).handle();
    }
    throw nb::type_error("expected moto.expr or moto.var");
}
var get_var_handle(const nb::handle &h) {
    try {
        return expr_cast<sym>(get_expr_handle(h));
    } catch (const std::bad_cast &) {
        throw nb::type_error("expected moto.var");
    }
}
} // namespace moto

namespace {
moto::ineq_constr::box_bound_t cast_box_bound(const nb::handle &h) {
    using namespace moto;

    if (nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h)) {
        return nb::cast<scalar_t>(h);
    }
    if (nb::isinstance<sym>(h) || nb::hasattr(h, "__sym__"))
        return static_cast<const cs::SX &>(*get_var_handle(h));
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

moto::generic_cost::tracking_param cast_tracking_param(const nb::handle &h) {
    using namespace moto;
    if (nb::isinstance<sym>(h) || nb::hasattr(h, "__sym__"))
        return get_var_handle(h);
    if (nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h))
        return nb::cast<scalar_t>(h);
    try {
        return nb::cast<vector>(h);
    } catch (const nb::cast_error &) {
        auto values = nb::cast<std::vector<scalar_t>>(h);
        vector out(values.size());
        for (size_t i = 0; i < values.size(); ++i)
            out(static_cast<Eigen::Index>(i)) = values[i];
        return out;
    }
}

using py_remap = std::vector<
    std::pair<moto::py_var_inarg_wrapper, moto::py_var_inarg_wrapper>>;

moto::generic_func::symbol_remap cast_remap(const py_remap &remap) {
    moto::generic_func::symbol_remap result;
    result.reserve(remap.size());
    for (const auto &[from, to] : remap)
        result.emplace_back(moto::var((moto::sym &)from),
                            moto::var((moto::sym &)to));
    return result;
}

std::shared_ptr<moto::generic_func> ready_func(moto::expr_handle handle) {
    auto result = moto::expr_cast<moto::generic_func>(handle);
    if (!result->finalized() && !result->finalize())
        throw std::runtime_error(
            fmt::format("function {} could not be finalized", result->name()));
    if (!result->wait_until_ready())
        throw std::runtime_error(
            fmt::format("function {} is not ready", result->name()));
    return result;
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
        return try_handle_cast([&] {
            for (auto &ex : l)
                value.emplace_back(*moto::get_var_handle(ex));
        });
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
        .def("clone", nb::overload_cast<const std::string &>(&sym::clone, nb::const_),
             nb::arg("name"), "Clone into an independent symbol with a fresh uid")
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
             [](generic_func &self, const py_remap &remap) {
                 return ready_func(self.remap_arguments(cast_remap(remap)));
             }, nb::arg("remap"), "Create a fresh remapped function")
        .def("reuse_remap",
             [](generic_func &self, const py_remap &remap) {
                 return ready_func(self.reuse_remap(cast_remap(remap)));
             }, nb::arg("remap"), "Reuse the cached function for this remap");

    nb::class_<generic_constr, generic_func>(m, "constr")
        .def_static(
            "create",
            [](const std::string &name, const cs::SX &out, approx_order order, field_t field) {
                return std::make_shared<generic_constr>(name, out, order, field);
            },
            nb::arg("name"), nb::arg("out"), nb::arg("order") = approx_order::first,
            nb::arg("field") = field_t::__undefined)
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
            [](const std::string &name, const cs::SX &out, approx_order order, field_t field) {
                return std::shared_ptr<generic_constr>(ineq_constr::create(
                    name, var_inarg_list{}, out, order, field));
            },
            nb::arg("name"), nb::arg("out"), nb::arg("order") = approx_order::first,
            nb::arg("field") = field_t::__undefined)
        .def_static(
            "create",
            [](const std::string &name, approx_order order, size_t dim, field_t field) {
                return std::shared_ptr<generic_constr>(ineq_constr::create(name, order, dim, field));
            },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd, nb::arg("field") = field_t::__undefined)
        .def_static(
            "create",
            [](const std::string &name,
               const cs::SX &out,
               const nb::handle &lb,
               const nb::handle &ub,
               approx_order order,
               field_t field) {
                return std::shared_ptr<generic_constr>(ineq_constr::create(
                    name, var_inarg_list{}, out, cast_box_bound(lb),
                    cast_box_bound(ub), order, field));
            },
            nb::arg("name"), nb::arg("out"), nb::arg("lb"), nb::arg("ub"),
            nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined)
        .def_static(
            "bounds",
            [](const std::string &name,
               const py_var_inarg_wrapper &value,
               const nb::handle &lb,
               const nb::handle &ub,
               approx_order order,
               field_t field) {
                var v((sym &)value);
                var_inarg_list args{*v};
                for (const nb::handle &bound : {lb, ub}) {
                    if (nb::isinstance<sym>(bound) || nb::hasattr(bound, "__sym__"))
                        args.emplace_back(*get_var_handle(bound));
                }
                return std::shared_ptr<generic_constr>(ineq_constr::create(
                    name, args, v, cast_box_bound(lb), cast_box_bound(ub), order, field));
            },
            nb::arg("name"), nb::arg("value"), nb::arg("lb"), nb::arg("ub"),
            nb::arg("order") = approx_order::first, nb::arg("field") = field_t::__undefined,
            "Create scalar or vector bounds directly on a variable");

    nb::class_<moto::pmm_constr, generic_constr>(m, "pmm_constr")
        .def_rw("rho", &moto::pmm_constr::rho, "Dual penalty weight for the proximal multiplier method");

    nb::class_<generic_cost, generic_func>(m, "cost")
        .def_static(
            "from_vector",
            [](const std::string &name, const cs::SX &value,
               const nb::handle &weight, const nb::handle &reference) {
                return std::shared_ptr<generic_cost>(generic_cost::from_vector(
                    name, var_inarg_list{}, value, cast_tracking_param(weight),
                    cast_tracking_param(reference)));
            },
            nb::arg("name"), nb::arg("value"), nb::arg("weight") = 1.0,
            nb::arg("reference") = 0.0)
        .def_static(
            "from_scalar",
            [](const std::string &name, const cs::SX &value,
               const nb::handle &weight, const nb::handle &reference) {
                return std::shared_ptr<generic_cost>(generic_cost::from_scalar(
                    name, var_inarg_list{}, value, cast_tracking_param(weight),
                    cast_tracking_param(reference)));
            },
            nb::arg("name"), nb::arg("value"), nb::arg("weight") = 1.0,
            nb::arg("reference") = 0.0)
        .def_prop_ro("weight", &generic_cost::weight)
        .def_prop_ro("reference", &generic_cost::reference);

    nb::class_<dense_dynamics, generic_constr>(m, "dense_dynamics")
        .def_static(
            "create",
            [](const std::string &name, const cs::SX &out, approx_order order) {
                return std::make_shared<dense_dynamics>(name, out, order);
            },
            nb::arg("name"), nb::arg("out"), nb::arg("order") = approx_order::first)
        .def_static(
            "create",
            [](const std::string &name, approx_order order, size_t dim) {
                return std::make_shared<dense_dynamics>(name, order, dim);
            },
            nb::arg("name"), nb::arg("order") = approx_order::first, nb::arg("dim") = dim_tbd)
        .def("mark_shared_inputs", &dense_dynamics::mark_shared_inputs, nb::arg("shared_inputs"));

    nb::class_<semi_implicit_euler, generic_constr>(m, "semi_implicit_euler")
        .def_static(
            "create",
            [](const std::string &name, const cs::SX &out,
               approx_order order) {
                return std::make_shared<semi_implicit_euler>(name, out, order);
            },
            nb::arg("name"), nb::arg("out"),
            nb::arg("order") = approx_order::first)
        .def("mark_shared_inputs", &semi_implicit_euler::mark_shared_inputs,
             nb::arg("shared_inputs"));
}
