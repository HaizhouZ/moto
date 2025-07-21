#ifndef MOTO_OCP_USR_FUNC_HPP
#define MOTO_OCP_USR_FUNC_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief user function pointer wrapper
 */
struct usr_func : public custom_func_derived<usr_func> {
    static auto *create(const std::string &name, approx_order order, size_t dim = dim_tbd) {
        return base::create(name, order, dim, __usr_func);
    }
    static auto *create(const std::string &name,
                        sym_init_list in_args, const cs::SX &out,
                        approx_order order = approx_order::first) {
        return base::create(name, in_args, out, order, __usr_func);
    }
};
} // namespace moto

#endif // MOTO_OCP_PRECOMP_HPP