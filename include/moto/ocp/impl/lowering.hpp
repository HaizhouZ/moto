#ifndef MOTO_OCP_IMPL_LOWERING_HPP
#define MOTO_OCP_IMPL_LOWERING_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto::lowering {

inline bool has_u_arg(const generic_func &func) {
    return func.has_u_arg();
}

inline bool has_pure_x_primal_args(const generic_func &func) {
    return func.has_pure_x_primal_args();
}

} // namespace moto::lowering

#endif
