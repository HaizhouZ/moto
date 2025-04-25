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
    dyn,    // dynamic model
    cost,   // "running cost"
    eq_constr, // "equality constraints"
    ineq_constr, // "inequality constraints"
};
constexpr size_t num_sym = p + 1;
constexpr size_t num = magic_enum::enum_count<type>();
constexpr size_t num_func = num - num_sym;
} // namespace field

} // namespace atri

#endif /*__FIELDS_*/