#ifndef ATRI_OCP_USR_FUNC_HPP
#define ATRI_OCP_USR_FUNC_HPP

#include <atri/ocp/func.hpp>

namespace atri {
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function helper class
 */
struct usr_func_impl : public func_impl {
    usr_func_impl(const std::string &name, approx_order order, size_t dim = 0)
        : func_impl(name, order, dim, __usr_func) {
    }
};
def_ptr(usr_func_impl);
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function pointer wrapper
 */
struct usr_func : public usr_func_impl_ptr_t {
    usr_func(const std::string &name, approx_order order, size_t dim = 0)
        : usr_func_impl_ptr_t(new usr_func_impl(name, order, dim)) {
    }
    usr_func(usr_func_impl *impl)
        : usr_func_impl_ptr_t(impl) {
    }
};
} // namespace atri

#endif // ATRI_OCP_PRECOMP_HPP