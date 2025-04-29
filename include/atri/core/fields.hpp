#ifndef __OPT_FIELDS__
#define __OPT_FIELDS__

#include <atri/core/fwd.hpp>

namespace atri {
namespace field {
enum type : size_t {
    x = 0,
    y,
    u,
    p,
    dyn,           // dynamic model
    cost,          // "running cost"
    eq_cstr_s,   // "state equality constraints"
    eq_cstr_c,   // "input-state equality constraints"
    ineq_cstr_x, // "state inequality constraints"
    ineq_cstr_u, // "state-input inequality constraints"
};
constexpr size_t num_sym = p + 1;
constexpr size_t num = magic_enum::enum_count<type>();
constexpr size_t num_func = num - num_sym;
} // namespace field

} // namespace atri

#endif /*__FIELDS_*/