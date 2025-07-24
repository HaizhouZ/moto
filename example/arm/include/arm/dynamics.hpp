#ifndef __MOTO_EXAMPLE_ARM_DYNAMICS_HPP__
#define __MOTO_EXAMPLE_ARM_DYNAMICS_HPP__

#include <moto/ocp/constr.hpp>

namespace moto {
struct armDynamics {
    sym q, v, tau, q_next, v_next;
    constr dyn_pos, dyn_vel;
    armDynamics() {
        std::tie(q, q_next) = sym::states("q", 7);
        std::tie(v, v_next) = sym::states("v", 7);
        tau = sym::inputs("tau", 7);
        std::cout << q << std::endl;
        std::cout << v << std::endl;
        std::cout << q_next << std::endl;
        std::cout << v_next << std::endl;
        std::cout << tau << std::endl;

        dyn_pos = constr("euler", approx_order::first, 7);
        dyn_pos.add_arguments({q, q_next, v_next});
        dyn_pos.field_hint().is_eq = true;

        std::cout << q << std::endl;
        std::cout << v << std::endl;
        std::cout << q_next << std::endl;
        std::cout << v_next << std::endl;
        std::cout << tau << std::endl;
        // dyn_pos.load_external();
        dyn_vel = constr("rnea", approx_order::first, 7);
        dyn_vel.add_arguments({q, v, v_next, tau});
        dyn_vel.field_hint().is_eq = true;
        // dyn_vel.load_external();
    }
};
} // namespace moto

#endif