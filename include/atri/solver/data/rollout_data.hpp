#ifndef __ROLLOUT_DATA__
#define __ROLLOUT_DATA__

#include <atri/core/array.hpp>
#include <atri/core/fields.hpp>

namespace atri {
namespace ns_riccati {
struct rollout_data {
    array<vector, field::num_sym> prim_;
    shifted_array<vector, field::num_constr, __dyn> dual_; // exclude cost
};
} // namespace ns_riccati

} // namespace atri

#endif