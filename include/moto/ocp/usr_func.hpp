#ifndef MOTO_OCP_USR_FUNC_HPP
#define MOTO_OCP_USR_FUNC_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function pointer wrapper
 */
struct usr_func : public impl::shared_handle<impl::custom_func, usr_func> {
    usr_func(const std::string &name, approx_order order, size_t dim = 0)
        : shared_handle(new expr_type(name, order, dim, __usr_func)) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRECOMP_HPP