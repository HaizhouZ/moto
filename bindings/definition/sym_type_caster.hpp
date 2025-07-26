#include <binding_fwd.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/ocp/usr_func.hpp>
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
        } else if (nb::isinstance<moto::var>(src)) {
            value = &static_cast<moto::sym &>(nb::cast<moto::var &>(src));
        } else if (nb::hasattr(src, "sym_base")) {
            value = &static_cast<moto::sym &>(nb::cast<moto::var &>(src.attr("sym_base")));
        } else {
            return false;
        }
#ifdef PRINT_SYM_CAST_INFO
        fmt::print("moto::sym use count {}\n", value->use_count());
        fmt::print("sym::impl use count {}\n", value->impl_use_count());
#endif
        return true;
    }

    // static handle from_cpp(const moto::sym &src, rv_policy policy, cleanup_list *cleanup) {
    //     // nb::print("Converting to Python from moto::sym\n");
    //     return nb::cast(moto::var(src));
    // }
};
// /// @brief Type caster for moto::sym
// template <>
// struct type_caster<moto::sym> {
//     NB_TYPE_CASTER(moto::sym, const_name("moto.sym"));

//     bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
// // Logic to convert Python object (src) to MyCustomType
// // e.g., extract an int from src and set self->value
// #ifdef PRINT_SYM_CAST_INFO

//         nb::print("Converting from Python to moto::sym");
// #endif
//         if (nb::isinstance<moto::shared_expr>(src)) {
//             value = static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src));
//         } else if (nb::hasattr(src, "sym_base")) {
//             value = static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(src.attr("sym_base")));
//         } else {
//             return false;
//         }
// #ifdef PRINT_SYM_CAST_INFO

//         fmt::print("moto::sym use count {}\n", value.use_count());
//         fmt::print("sym::impl use count {}\n", value.impl_use_count());
// #endif
//         return true;
//     }

//     static handle from_cpp(const moto::sym &src, rv_policy policy, cleanup_list *cleanup) {
//         nb::print("Converting to Python from moto::sym\n");
//         return nb::cast(moto::shared_expr(src));
//     }
// };
} // namespace detail
} // namespace nanobind

namespace moto {

// using expr_in_list = moto::expr_list;

struct expr_in_list : public std::vector<nb::handle> {
    // auto operator*() const {
    //     std::vector<std::reference_wrapper<sym>> tmp;
    //     tmp.reserve(this->size());
    //     for (auto &ex : *this) {
    //         if (nb::isinstance<moto::shared_expr>(ex)) {
    //             tmp.push_back(static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(ex)));
    //         } else if (nb::hasattr(ex, "sym_base")) {
    //             tmp.push_back(static_cast<moto::sym &>(nb::cast<moto::shared_expr &>(ex.attr("sym_base"))));
    //         }
    //     }
    //     return tmp;
    // }
    operator expr_list() const;
    // operator
};
} // namespace moto

namespace nanobind {
namespace detail {
template <>
struct type_caster<moto::expr_list> {
    NB_TYPE_CASTER(moto::expr_list, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                                        const_name("[") + make_caster<moto::shared_expr>::Name +
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
            } else if (type.is(nb::type<moto::expr>())) {
                value.push_back(nb::cast<moto::expr &>(ex));
            // } else if (type.is(nb::type<moto::func_base>())) {
            //     value.push_back(nb::cast<moto::func_base &>(ex));
            // } else if (type.is(nb::type<moto::cost>())) {
            //     value.push_back(nb::cast<moto::cost &>(ex));
            // } else if (type.is(nb::type<moto::constr>())) {
            //     value.push_back(nb::cast<moto::constr &>(ex));
            // } else if (type.is(nb::type<moto::pre_compute>())) {
            //     value.push_back(nb::cast<moto::pre_compute &>(ex));
            // } else if (type.is(nb::type<moto::usr_func>())) {
            //     value.push_back(nb::cast<moto::usr_func &>(ex));
            // } else if (type.is(nb::type<moto::custom_func>())) {
            //     value.push_back(nb::cast<moto::custom_func &>(ex));
            } else {
                nb::print("Unknown type in var_list caster: ", ex);
            }
        }
        return true;
    }

    static handle from_cpp(const moto::expr_list &src, rv_policy policy, cleanup_list *cleanup) {
        nb::list l;
        for (auto &ex : src) {
            l.append(std::move(ex));
        }
        return l.release();
    }
};
template <>
struct type_caster<moto::expr_in_list> : public list_caster<moto::expr_in_list, handle> {};
} // namespace detail
} // namespace nanobind
