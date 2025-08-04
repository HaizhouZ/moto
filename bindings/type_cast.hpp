#include <binding_fwd.hpp>
/// @brief A helper struct to extract the PySwigObject from a Python object
/// This is used to access the underlying C++ object pointer from a Python object
/// that has been wrapped with SWIG.
/// The PySwigObject struct is expected to have a 'this' attribute that points to
/// the C++ object pointer.
/// The 'desc' attribute is used to store the type description of the object.
/// @cite https://docs.ros.org/en/noetic/api/eigenpy/html/swig_8hpp_source.html
struct PySwigObject {
    PyObject_HEAD void *ptr;
    const char *desc;
};

inline PySwigObject *get_PySwigObject(PyObject *pyObj) {
    if (!PyObject_HasAttrString(pyObj, "this"))
        return NULL;

    PyObject *this_ptr = PyObject_GetAttrString(pyObj, "this");
    if (this_ptr == NULL)
        return nullptr;
    PySwigObject *swig_obj = reinterpret_cast<PySwigObject *>(this_ptr);

    return swig_obj;
}

#include <casadi/casadi.hpp>

namespace cs = casadi;

namespace nanobind {
namespace detail {
template <>
struct type_caster<cs::SX> {
    NB_TYPE_CASTER(cs::SX, /* type_name_for_error_messages */ const_name("casadi.SX"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        // Logic to convert Python object (src) to MyCustomType
        // e.g., extract an int from src and set self->value
        value = cs::SX();
        auto swig_obj = get_PySwigObject(src.ptr());
        if (swig_obj) {
            assert(swig_obj != nullptr);
            auto sx_ptr = reinterpret_cast<cs::SX *>(swig_obj->ptr);
            value = *sx_ptr;
        } else
            nb::raise_type_error("Expected a casadi.SX object, but got: %s", nb::type_name(src).c_str());
        return true;
    }

    static handle from_cpp(const cs::SX &src, rv_policy policy, cleanup_list *cleanup) {
        // Logic to convert MyCustomType (src) to a Python object
        // e.g., return a Python int from src.value
        object py_cs_module = nb::module_::import_("casadi");
        object py_cs_sx = py_cs_module.attr("SX")();
        auto swig_obj = get_PySwigObject(py_cs_sx.ptr());
        assert(swig_obj != nullptr);
        auto sx_ptr = reinterpret_cast<cs::SX *>(swig_obj->ptr);
        *sx_ptr = src;
        return py_cs_sx.release();
    }
};
} // namespace detail
} // namespace nanobind

#include <moto/core/array.hpp>
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

#include <moto/ocp/impl/func.hpp>

namespace moto {
shared_expr &cast_to_shared_expr(const nb::handle &h);
var &cast_to_var(const nb::handle &h);
func &cast_to_func(const nb::handle &h);
} // namespace moto

namespace nanobind {
namespace detail {
template <typename T>
struct type_caster<moto::impl::unique_id<T>> {
    NB_TYPE_CASTER(moto::impl::unique_id<T>, const_name("int"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        value = size_t(src);
        return true;
    }

    static handle from_cpp(const moto::impl::unique_id<T> &src, rv_policy policy, cleanup_list *cleanup) {
        return nb::int_(size_t(src));
    }
};
template <>
struct type_caster<moto::expr_inarg_list> {
    NB_TYPE_CASTER(moto::expr_inarg_list, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                                              const_name("[") +
                                              make_caster<moto::var>::Name +
                                              const_name(" | ") +
                                              make_caster<moto::func>::Name +
                                              const_name(" | casadi.SX]"))

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
                value.push_back(moto::cast_to_shared_expr(ex));
            } catch (const std::exception &e) {
                fmt::print("Failed to cast to shared_expr: {}\n", e.what());
                return false;
            }
        }
        return true;
    }
};
} // namespace detail
} // namespace nanobind

namespace moto {
struct py_var_wrapper {
    moto::var *v = nullptr;
    py_var_wrapper() = default;
    py_var_wrapper(moto::var &vref) : v(&vref) {}
    py_var_wrapper(const py_var_wrapper &rhs) = default;
    py_var_wrapper(py_var_wrapper &&rhs) noexcept = default;
    py_var_wrapper &operator=(const py_var_wrapper &rhs) = default;
    py_var_wrapper &operator=(py_var_wrapper &&rhs) noexcept = default;
    operator moto::var &() const { return *v; }
    operator moto::sym &() const { return *v; }
};
struct py_shared_expr_wrapper {
    moto::shared_expr *v = nullptr;
    py_shared_expr_wrapper() = default;
    py_shared_expr_wrapper(moto::shared_expr &vref) : v(&vref) {}
    py_shared_expr_wrapper(const py_shared_expr_wrapper &rhs) = default;
    py_shared_expr_wrapper(py_shared_expr_wrapper &&rhs) noexcept = default;
    py_shared_expr_wrapper &operator=(const py_shared_expr_wrapper &rhs) = default;
    py_shared_expr_wrapper &operator=(py_shared_expr_wrapper &&rhs) noexcept = default;
    operator moto::shared_expr &() const { return *v; }
};
} // namespace moto
namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::py_var_wrapper> {
    NB_TYPE_CASTER(moto::py_var_wrapper, const_name("moto.var | casadi.SX"));
    bool from_python(handle src, uint8_t flags, void *ptr) {
        try {
            value = std::move(moto::py_var_wrapper(moto::cast_to_var(src)));
        } catch (const std::exception &e) {
            fmt::print("Failed to cast to moto.var: {}\n", e.what());
            return false;
        }
        return true;
    }
};
template <>
struct type_caster<moto::py_shared_expr_wrapper> {
    NB_TYPE_CASTER(moto::py_shared_expr_wrapper, const_name("moto.shared_expr | casadi.SX"));
    bool from_python(handle src, uint8_t flags, void *ptr) {
        value = std::move(moto::py_shared_expr_wrapper(moto::cast_to_shared_expr(src)));
        return true;
    }
};
} // namespace detail
} // namespace nanobind