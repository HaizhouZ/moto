#ifndef __OPT_FIELDS__
#define __OPT_FIELDS__

#include <moto/core/fwd.hpp>
#include <magic_enum/magic_enum.hpp>

namespace moto {
enum field_t : size_t {
    __x = 0,
    __y,
    __u,
    __p,          // non-decision parameters
    __dyn,        // dynamic model
    __eq_x,       // "state equality constraints"
    __eq_xu,      // "input-state equality constraints"
    __ineq_x,     // "state inequality constraints"
    __ineq_xu,    // "state-input inequality constraints"
    __eq_x_soft,  // "soft state equality constraints"
    __eq_xu_soft, // "soft state-input equality constraints"
    __cost,       // "running cost"
    __usr_var,    // user defined variables
    // user functions
    __pre_comp, // will be called before the computation of the approximation update, also the data will be shared
    __usr_func, // user function, will not be called in the approximation update, only users can use them
    NUM,
    __undefined,
};

constexpr auto state_fields = std::array{__x, __y};
constexpr auto nonstate_fields = std::array{__u};
constexpr auto primal_fields = std::array{__x, __y, __u};
constexpr auto hard_constr_fields = std::array{__dyn, __eq_x, __eq_xu};
constexpr auto hard_constr_fields_non_dyn = std::array{__eq_x, __eq_xu};
constexpr auto ineq_constr_fields = std::array{__ineq_x, __ineq_xu};
constexpr auto soft_constr_fields = std::array{__eq_x_soft, __eq_xu_soft};

template <size_t N, typename T>
inline bool in_field(T val, const std::array<field_t, N> &arr) {
    return std::find(arr.begin(), arr.end(), field_t(val)) != arr.end();
}

template <std::size_t... sizes>
constexpr auto concat_fields(const std::array<field_t, sizes> &...arrays) {
    std::array<field_t, (sizes + ...)> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), sizes, result.begin() + index), index += sizes), ...);

    return result;
}

constexpr auto ineq_soft_constr_fields = concat_fields(ineq_constr_fields, soft_constr_fields);
constexpr auto constr_fields = concat_fields(hard_constr_fields, ineq_soft_constr_fields);
constexpr auto func_fields = concat_fields(constr_fields, std::array{__cost});
constexpr auto custom_func_fields = std::array{__pre_comp, __usr_func};

namespace field {
constexpr auto name(field_t f) { return magic_enum::enum_name<field_t>(f); }
constexpr size_t num_sym = __p + 1;                   // number of symbolic fields
constexpr size_t num_prim = std::size(primal_fields); // number of primal fields
constexpr size_t num = NUM;
constexpr size_t num_func = __cost + 1 - num_sym;
constexpr size_t num_constr = num_func - 1; // exclude cost
} // namespace field

constexpr inline auto format_as(field_t f) { return field::name(f); }
} // namespace moto

#endif /*__FIELDS_*/