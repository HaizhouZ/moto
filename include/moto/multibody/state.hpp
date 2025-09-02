#ifndef MOTO_MULTIBODY_STATE_HPP
#define MOTO_MULTIBODY_STATE_HPP

#include <moto/multibody/fwd.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {
namespace multibody {
struct state {
    var q, v, a, dt; // position, velocity, acceleration, timestep
    root_joint_t root_type;
};
} // namespace multibody
} // namespace moto

#endif