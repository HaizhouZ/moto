#ifndef MOTO_MULTIBODY_FWD_HPP
#define MOTO_MULTIBODY_FWD_HPP

#include <cstddef>

namespace moto {
namespace multibody {

    enum class root_joint_t : size_t {
    xyz_quat = 0,
    xyz_eulerZYX,
    fixed
};

}
} // namespace moto

#endif // MOTO_MULTIBODY_FWD_HPP