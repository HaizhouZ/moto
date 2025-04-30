#ifndef __ROLLOUT_DATA__
#define __ROLLOUT_DATA__

#include <atri/core/fields.hpp>

namespace atri {
struct rollout_data {
    vector prim_[field::num_sym];
    vector dual_[field::num_constr]; // exclude cost
};

} // namespace atri

#endif