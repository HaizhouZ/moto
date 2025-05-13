#ifndef __OPT_FIELDS__
#define __OPT_FIELDS__

#include <atri/core/fwd.hpp>

namespace atri {
enum field_t : size_t {
    __x = 0,
    __u,
    __y,
    __p,         // parameters
    __dyn,       // dynamic model
    __eq_cstr_s, // "state equality constraints"
    __eq_cstr_c, // "input-state equality constraints"
    __cost,      // "running cost"
    // ineq_cstr_x, // "state inequality constraints"
    // ineq_cstr_u, // "state-input inequality constraints"
    NUM,
};
namespace field {
constexpr size_t num_sym = __p + 1;  // number of symbolic fields
constexpr size_t num_prim = __y + 1; // number of primal fields
constexpr size_t num = NUM;
constexpr size_t num_func = num - num_sym;
constexpr size_t num_constr = num_func - 1;
;
} // namespace field

} // namespace atri

#endif /*__FIELDS_*/