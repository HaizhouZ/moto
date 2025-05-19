#ifndef __ATRI_EXAMPLE_ARM_DYNAMICS_HPP__
#define __ATRI_EXAMPLE_ARM_DYNAMICS_HPP__

#include <atri/core/external_function.hpp>
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

        dyn_pos = constr("arm_dynamics_pos", 7, __dyn, approx_order::first);
        dyn_pos->add_arguments({q, q_next, v_next});
        dyn_pos->value = [](sparse_approx_data &data) {
            data.v_ = data.in_args_[0] - data.in_args_[1] + 0.01 * data.in_args_[2];
        };
        dyn_pos->jacobian = [](sparse_approx_data &data) {
            data.jac_[0].diagonal().setConstant(-1);
            data.jac_[1].setIdentity();
            data.jac_[2].diagonal().setConstant(0.01);
        };
        dyn_vel = constr("arm_dynamics_vel", 7, __dyn, approx_order::first);
        dyn_vel->add_arguments({v, v_next, tau});
        dyn_vel->value = [](sparse_approx_data &data) {
            static ext_func rnea("gen/librnea.so", "rnea");
            rnea.invoke(data.in_args_, data.v_);
        };
        dyn_vel->jacobian = [](sparse_approx_data &data) {
            static ext_func rnea_jac("gen/librnea_jac.so", "rnea_jac");
            rnea_jac.invoke(data.in_args_, data.jac_);
        };

        dyn_pos->add_arguments({q, q_next, v_next});
        dyn_vel->add_arguments({q, v, v_next, tau});
    }
};
} // namespace atri

#endif