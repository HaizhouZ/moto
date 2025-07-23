#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function pointer wrapper
 */
struct pre_compute : public custom_func {
    pre_compute(const std::string &name) : custom_func(name, approx_order::none, 0, __pre_comp) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP