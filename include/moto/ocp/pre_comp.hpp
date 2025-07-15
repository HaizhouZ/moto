#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {
namespace impl {
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function helper class
 */
struct pre_compute : public impl::func {
    pre_compute(const std::string &name)
        : impl::func(name, __pre_comp) {
    }
};
} // namespace impl
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function pointer wrapper
 */
struct pre_compute : public impl::shared_<impl::pre_compute, pre_compute> {
    pre_compute(const std::string &name)
        : shared_(new expr_type(name)) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP