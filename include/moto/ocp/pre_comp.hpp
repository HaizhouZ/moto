#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function pointer wrapper
 */
struct pre_compute : public custom_func_derived<pre_compute> {
    static auto *create(const std::string &name) {
        return base::create(name, approx_order::none, 0, __pre_comp);
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP