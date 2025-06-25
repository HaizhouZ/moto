#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function helper class
 */
struct pre_compute_impl : public func_impl {
    pre_compute_impl(const std::string &name)
        : func_impl(name, __pre_comp) {
    }
};
def_ptr(pre_compute_impl);
/////////////////////////////////////////////////////////////////////
/**
 * @brief pre-compute function pointer wrapper
 */
struct pre_compute : public pre_compute_impl_ptr_t {
    pre_compute(const std::string &name)
        : pre_compute_impl_ptr_t(new pre_compute_impl(name)) {
    }
    pre_compute(pre_compute_impl *impl)
        : pre_compute_impl_ptr_t(impl) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP