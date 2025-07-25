#include <binding_fwd.hpp>
#include <moto/ocp/sym.hpp>

// #define PRINT_SYM_CAST_INFO

namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::sym *> {
    NB_TYPE_CASTER(moto::sym *, const_name("moto.sym"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
        // Logic to convert Python object (src) to MyCustomType
        // e.g., extract an int from src and set self->value
#ifdef PRINT_SYM_CAST_INFO
        nb::print("Converting from Python to moto::sym* (ptr_t)");
#endif
        if (nb::isinstance<moto::shared_expr>(src)) {
            value = &static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src));
        } else if (nb::hasattr(src, "sym_base")) {
            value = &static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src.attr("sym_base")));
        } else {
            return false;
        }
#ifdef PRINT_SYM_CAST_INFO
        fmt::print("moto::sym use count {}\n", value->use_count());
        fmt::print("sym::impl use count {}\n", value->impl_use_count());
#endif
        return true;
    }

    static handle from_cpp(const moto::sym &src, rv_policy policy, cleanup_list *cleanup) {
        // nb::print("Converting to Python from moto::sym\n");
        return nb::cast(moto::shared_expr(src));
    }
};
/// @brief Type caster for moto::sym
template <>
struct type_caster<moto::sym> {
    NB_TYPE_CASTER(moto::sym, const_name("moto.sym"));

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
// Logic to convert Python object (src) to MyCustomType
// e.g., extract an int from src and set self->value
#ifdef PRINT_SYM_CAST_INFO

        nb::print("Converting from Python to moto::sym");
#endif
        if (nb::isinstance<moto::shared_expr>(src)) {
            value = static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src));
        } else if (nb::hasattr(src, "sym_base")) {
            value = static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src.attr("sym_base")));
        } else {
            return false;
        }
#ifdef PRINT_SYM_CAST_INFO

        fmt::print("moto::sym use count {}\n", value.use_count());
        fmt::print("sym::impl use count {}\n", value.impl_use_count());
#endif
        return true;
    }

    static handle from_cpp(const moto::sym &src, rv_policy policy, cleanup_list *cleanup) {
        nb::print("Converting to Python from moto::sym\n");
        return nb::cast(moto::shared_expr(src));
    }
};
} // namespace detail
} // namespace nanobind

namespace moto {

struct sym_in_list : public std::vector<nb::handle> {
    auto operator*() const {
        std::vector<std::reference_wrapper<sym>> tmp;
        tmp.reserve(this->size());
        for (auto &ex : *this) {
            if (nb::isinstance<moto::shared_expr>(ex)) {
                tmp.push_back(static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(ex)));
            } else if (nb::hasattr(ex, "sym_base")) {
                tmp.push_back(static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(ex.attr("sym_base"))));
            }
        }
        return tmp;
    }
    operator expr_list() const;
};
} // namespace moto

namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::sym_list> : public list_caster<moto::sym_list, moto::sym> {};
template <>
struct type_caster<moto::sym_in_list> : public list_caster<moto::sym_in_list, handle> {};
} // namespace detail
} // namespace nanobind
