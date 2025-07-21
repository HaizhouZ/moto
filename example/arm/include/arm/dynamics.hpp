#ifndef __MOTO_EXAMPLE_ARM_DYNAMICS_HPP__
#define __MOTO_EXAMPLE_ARM_DYNAMICS_HPP__

#include <moto/ocp/constr.hpp>

template <typename T>
using ref_ = std::add_lvalue_reference_t<T>;
template <typename T>
using ptr_ = T*;


namespace moto {
struct armDynamics {
    sym_ptr_t q, q_next, v, v_next, tau;
    constr_ptr_t dyn_pos, dyn_vel;
    armDynamics() : q(sym::state("q", 7)), q_next(q->next()), v(sym::state("v", 7)), v_next(v->next()),
                    tau(sym::inputs("tau", 7)),
                    dyn_pos(constr::create("euler", approx_order::first, 7)),
                    dyn_vel(constr::create("rnea", approx_order::first, 7)) {

        dyn_pos->add_arguments({q, q_next, v_next});
        dyn_pos->load_external();
        dyn_vel->add_arguments({q, v, v_next, tau});
        dyn_vel->load_external();
    }
};
} // namespace moto

#endif