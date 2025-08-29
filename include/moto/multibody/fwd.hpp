#ifndef MOTO_MULTIBODY_FWD_HPP
#define MOTO_MULTIBODY_FWD_HPP

#include <pinocchio/multibody/fwd.hpp>

namespace moto {

namespace pin = pinocchio;
using pin_model = pin::Model;
using pin_data = pin::Data;
}; // namespace moto

enum class root_joint_t : size_t {
    xyz_quat = 0,
    xyz_eulerZYX,
    fixed
};

#endif // MOTO_MULTIBODY_FWD_HPP