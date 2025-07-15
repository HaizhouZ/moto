#ifndef MOTO_OCP_USR_FUNC_HPP
#define MOTO_OCP_USR_FUNC_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function helper class
 */
namespace impl {
struct usr_func : public func {
    usr_func(const std::string &name, approx_order order, size_t dim = 0)
        : func(name, order, dim, __usr_func) {
    }
};
}
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function pointer wrapper
 */
struct usr_func : public impl::shared_<impl::usr_func, usr_func> {
    usr_func(const std::string &name, approx_order order, size_t dim = 0)
        : shared_(new expr_type(name, order, dim)) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRECOMP_HPP