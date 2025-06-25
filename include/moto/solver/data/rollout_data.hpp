#ifndef __ROLLOUT_DATA__
#define __ROLLOUT_DATA__

#include <moto/core/array.hpp>
#include <moto/core/fields.hpp>

namespace moto {
namespace ns_riccati {
struct rollout_data {
    array<vector, field::num_sym> prim_;
    shifted_array<vector, field::num_constr, __dyn> dual_; // exclude cost
};
} // namespace ns_riccati

} // namespace moto

#endif