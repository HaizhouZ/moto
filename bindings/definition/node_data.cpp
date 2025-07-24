#include <binding_fwd.hpp>
#include <moto/ocp/impl/node_data.hpp>
#include <nanobind/stl/array.h>

namespace nanobind {
namespace detail {
template <typename Entry, size_t Size, size_t shift>
struct type_caster<moto::shifted_array<Entry, Size, shift>> {
    using Array = moto::shifted_array<Entry, Size, shift>;
    NB_TYPE_CASTER(Array, const_name("moto.shifted_array"))
    using Caster = make_caster<Entry>;

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        PyObject *temp;

        /* Will initialize 'temp' (NULL in the case of a failure.) */
        PyObject **o = seq_get_with_size(src.ptr(), Size, &temp);

        Caster caster;
        bool success = o != nullptr;

        flags = flags_for_local_caster<Entry>(flags);

        if (success) {
            for (size_t i = 0; i < Size; ++i) {
                if (!caster.from_python(o[i], flags, cleanup) ||
                    !caster.template can_cast<Entry>()) {
                    success = false;
                    break;
                }

                value[i] = caster.operator cast_t<Entry>();
            }

            Py_XDECREF(temp);
        }

        return success;
    }

    template <typename T>
    static handle from_cpp(T &&src, rv_policy policy, cleanup_list *cleanup) {
        object ret = steal(PyList_New(Size));

        if (ret.is_valid()) {
            Py_ssize_t index = 0;

            for (auto &value : src) {
                handle h = Caster::from_cpp(forward_like_<T>(value), policy, cleanup);

                if (!h.is_valid()) {
                    ret.reset();
                    break;
                }

                NB_LIST_SET_ITEM(ret.ptr(), index++, h.ptr());
            }
        }

        return ret.release();
    }
};

} // namespace detail
} // namespace nanobind

void register_submodule_node_data(nb::module_ &m) {
    using namespace moto;
    nb::class_<sym_data>(m, "sym_data")
        .def_prop_ro("prob", [](sym_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](sym_data &self, const sym &s) -> auto { return self.operator[](s); });
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
        .def_prop_ro("prob", [](shared_data &self) -> ocp & { return *self.prob_; })
        .def("__getitem__", [](shared_data &self, const std::shared_ptr<impl::func> &ptr) -> auto & { return self[ptr]; });
    nb::class_<node_data>(m, "node_data")
        .def_prop_ro("prob", [](node_data &self) -> auto & { return *self.prob_; })
        .def_prop_ro("sym", [](node_data &self) -> auto & { return *self.sym_; })
        .def_prop_ro("dense", [](node_data &self) -> auto & { return *self.dense_; })
        .def_prop_ro("shared", [](node_data &self) -> auto & { return *self.impl_; })
        .def("value", nb::overload_cast<field_t>(&node_data::value, nb::const_), nb::arg("field"),
             "Get value of the whole field")
        .def("value", nb::overload_cast<const sym &>(&node_data::value, nb::const_), nb::arg("sym"),
             "Get value of the sym variable")
        .def("data", &node_data::data<impl::func>, nb::arg("func"),
             "Get the sparse func data by pointer")
        .def("cost", &node_data::cost, "Get the cost value");
}