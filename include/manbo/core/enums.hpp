#ifndef __OPT_ENUMS__
#define __OPT_ENUMS__

#include <magic_enum.hpp>

namespace atri {
namespace __details {
namespace field {
enum type : size_t {
    x = 0,
    y,
    u,
    p,
    dyn,     // dynamic model
    cost,    // "running cost"
    constr,  // "constraints"
};
constexpr size_t num_sym = p + 1;
constexpr size_t num = magic_enum::enum_count<type>();
constexpr size_t num_func = num - num_sym;
}  // namespace field
}  // namespace __details

namespace field = __details::field;
typedef field::type field_type;

}  // namespace atri

#endif /*__ENUMS_*/