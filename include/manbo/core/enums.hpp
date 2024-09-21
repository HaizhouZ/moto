#ifndef __OPT_ENUMS__
#define __OPT_ENUMS__

namespace manbo {
namespace details {
namespace field {
enum type : size_t {
    x = 0,
    y,
    u,
    p,
    dyn,       // dynamic model
    cost,      // "running cost"
    constr,    // "constraints"
    num_field,
};
}

}  // namespace details
typedef details::field::type field_type;

}  // namespace manbo

#endif /*__ENUMS_*/