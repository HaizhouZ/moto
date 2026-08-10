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
        } else {
            return false;
        }
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

template <>
struct type_caster<cs::MX> {
    NB_TYPE_CASTER(cs::MX, const_name("casadi.MX"));

    bool from_python(handle src, uint8_t, cleanup_list *) {
        value = cs::MX();
        auto swig_obj = get_PySwigObject(src.ptr());
        if (!swig_obj) return false;
        value = *reinterpret_cast<cs::MX *>(swig_obj->ptr);
        return true;
    }

    static handle from_cpp(const cs::MX &src, rv_policy, cleanup_list *) {
        object module = nb::module_::import_("casadi");
        object result = module.attr("MX")();
        auto swig_obj = get_PySwigObject(result.ptr());
        assert(swig_obj != nullptr);
        *reinterpret_cast<cs::MX *>(swig_obj->ptr) = src;
        return result.release();
    }
};
} // namespace detail
} // namespace nanobind

#include <moto/ocp/impl/func.hpp>

namespace moto {
expr_handle get_expr_handle(const nb::handle &h);
var get_var_handle(const nb::handle &h);
} // namespace moto

namespace nanobind {
namespace detail {
template <typename F>
bool try_handle_cast(F &&cast) noexcept {
    try {
        cast();
        return true;
    } catch (...) {
        return false;
    }
}

template <typename T>
struct type_caster<moto::utils::unique_id<T>> {
    NB_TYPE_CASTER(moto::utils::unique_id<T>, const_name("int"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        value = size_t(src);
        return true;
    }

    static handle from_cpp(const moto::utils::unique_id<T> &src, rv_policy policy, cleanup_list *cleanup) {
        return nb::int_(size_t(src));
    }
};
template <>
struct type_caster<moto::expr_inarg_list> {
    NB_TYPE_CASTER(moto::expr_inarg_list, io_name("collections.abc.Sequence", "list") +
                                              const_name("[ moto.expr ") +
                                              const_name(" | moto.var]"))

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
                value.emplace_back(*moto::get_expr_handle(ex));
        });
    }
};
} // namespace detail
} // namespace nanobind

namespace moto {
/// @brief read-only inarg wrapper for moto::var
struct py_var_inarg_wrapper {
    var v;
    operator moto::sym &() const { return *v; }
};
} // namespace moto
namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::py_var_inarg_wrapper> {
    NB_TYPE_CASTER(moto::py_var_inarg_wrapper, const_name("moto.var"));
    bool from_python(handle src, uint8_t flags, void *ptr) {
        return try_handle_cast([&] {
            value.v = moto::get_var_handle(src);
        });
    }
};
/// @brief Type caster for an owning expression handle.
template <>
struct type_caster<moto::expr_handle> {
    NB_TYPE_CASTER(moto::expr_handle, const_name("moto.expr"));
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        return try_handle_cast([&] {
            value = moto::get_expr_handle(src);
        });
    }
    static handle from_cpp(const moto::expr_handle &src, rv_policy policy, cleanup_list *cleanup) {
        return type_caster<std::shared_ptr<moto::expr>>::from_cpp(src, policy, cleanup);
    }
};
/// @brief Type caster for moto::var
/// @note for inargs of funcs, use @ref moto::py_var_inarg_wrapper instead
template <>
struct type_caster<moto::var> {
    NB_TYPE_CASTER(moto::var, const_name("moto.var"));
    bool from_python(handle src, uint8_t flags, void *ptr) {
        return try_handle_cast([&] {
            value = moto::get_var_handle(src);
        });
    }
    static nb::handle from_cpp(const moto::var &src, rv_policy policy, cleanup_list *cleanup) {
        nb::object py_cs_module = nb::module_::import_("moto");
        nb::object py_cs_var = py_cs_module.attr("var")(type_caster<std::shared_ptr<moto::sym>>::from_cpp(src, policy, cleanup));
        return py_cs_var.release();
    }
};
} // namespace detail
} // namespace nanobind
