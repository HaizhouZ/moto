#ifndef __ATRI_EXAMPLE_ARM_DYNAMICS_HPP__
#define __ATRI_EXAMPLE_ARM_DYNAMICS_HPP__

#include <atri/ocp/constr.hpp>
#include <atri/ocp/dynamics.hpp>

namespace atri {
struct armDynamics {
    sym q, v, tau, q_next, v_next;
    constr dyn_pos, dyn_vel;
    armDynamics() {
        std::tie(q, q_next) = dynamics::make_state("q", 7);
        std::tie(v, v_next) = dynamics::make_state("v", 7);
        tau = dynamics::make_input("tau", 7);

        dyn_pos = constr("euler", approx_order::first, 7);
        dyn_pos->add_arguments({q, q_next, v_next});
        dyn_pos->load_external();
        dyn_vel = constr("rnea", approx_order::first, 7);
        dyn_vel->add_arguments({q, v, v_next, tau});
        dyn_vel->load_external();
    }
};
} // namespace atri

#endif