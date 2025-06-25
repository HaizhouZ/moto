#ifndef __OPT_FIELDS__
#define __OPT_FIELDS__

#include <moto/core/fwd.hpp>

namespace moto {
enum field_t : size_t {
    __x = 0,
    __u,
    __y,
    __p,         // non-decision parameters
    __dyn,       // dynamic model
    __eq_cstr_s, // "state equality constraints"
    __eq_cstr_c, // "input-state equality constraints"
    // ineq_cstr_x, // "state inequality constraints"
    // ineq_cstr_u, // "state-input inequality constraints"
    __cost,    // "running cost"
    __usr_var, // user defined variables
    // user functions
    __pre_comp, // will be called before the computation of the approximation update, also the data will be shared
    __usr_func, // user function, will not be called in the approximation update, only users can use them
    NUM,
    __undefined,
};
namespace field {
constexpr size_t num_sym = __p + 1;  // number of symbolic fields
constexpr size_t num_prim = __y + 1; // number of primal fields
constexpr size_t num = NUM;
constexpr size_t num_func = __cost + 1 - num_sym;
constexpr size_t num_constr = num_func - 1; // exclude cost
} // namespace field

} // namespace moto

#endif /*__FIELDS_*/