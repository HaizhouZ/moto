#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function pointer wrapper
 */
struct pre_compute : public impl::shared_handle<impl::custom_func, pre_compute> {
    pre_compute(const std::string &name)
        : shared_handle(new expr_type(name, __pre_comp)) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP