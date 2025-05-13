#ifndef __ROLLOUT_DATA__
#define __ROLLOUT_DATA__

#include <atri/core/fields.hpp>
#include <atri/core/offset_array.hpp>

namespace atri {
struct rollout_data {
    vector prim_[field::num_sym];
    offset_array<vector, field::num_constr, __dyn> dual_; // exclude cost
};

} // namespace atri

#endif