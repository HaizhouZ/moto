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
    // ineq_cstr_x, // "state inequality constraints"
    // ineq_cstr_u, // "state-input inequality constraints"
    __cost, // "running cost"
    NUM,
    // user functions
    __pre_comp, // will be called before the computation of the approximation update, also the data will be shared
    __usr_func, // user function, will not be called in the approximation update, only users can use them
    __undefined,
};
namespace field {
constexpr size_t num_sym = __p + 1;  // number of symbolic fields
constexpr size_t num_prim = __y + 1; // number of primal fields
constexpr size_t num = NUM;
constexpr size_t num_func = num - num_sym;
constexpr size_t num_constr = num_func - 1; // exclude cost
} // namespace field

} // namespace atri

#endif /*__FIELDS_*/