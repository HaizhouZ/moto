#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#include <atri/ocp/constr.hpp>
#include <atri/ocp/dynamics.hpp>

namespace atri {

/**
 * @brief double integrator dynamics
 * v_next - v - a * dt = 0
 * r_next - r - v_next * dt = 0
 */
class doubleIntegratorDyn : public dynamics, public collection {
  public:
    // position, velocity, acceleration, position, velocity
    sym r, v, a, r_next, v_next;
    constr dyn_pos, dyn_vel, vel_zero_constr;
    struct pos : public constr_impl {
        pos() : constr_impl("doubleIntegratorDynamics_pos", 3, __dyn, approx_order::first) {
            value = [](sparse_approx_data &data) {
                data.v_ = -data.in_args_[0] + data.in_args_[1] - 0.01 * data.in_args_[2];
            };
            jacobian = [](sparse_approx_data &data) {
                data.jac_[0].diagonal().setConstant(-1);
                data.jac_[1].setIdentity();
                data.jac_[2].diagonal().setConstant(-0.01);
            };
        }
    };
    struct vel : public constr_impl {
        vel() : constr_impl("doubleIntegratorDynamics_vel", 3, __dyn, approx_order::first) {
            value = [](sparse_approx_data &data) {
                data.v_ = -data.in_args_[0] + data.in_args_[1] - 0.01 * data.in_args_[2];
            };
            jacobian = [](sparse_approx_data &data) {
                data.jac_[0].diagonal().setConstant(-1);
                data.jac_[1].setIdentity();
                data.jac_[2].diagonal().setConstant(-0.01);
            };
        }
    };
    struct zero_vel: public constr_impl{
        zero_vel() : constr_impl("doubleIntegratorDynamics_zero_vel", 3, __eq_cstr_s, approx_order::first) {
            value = [](sparse_approx_data &data) {
                data.v_ = data.in_args_[0];
            };
            jacobian = [](sparse_approx_data &data) {
                data.jac_[0].setIdentity();
            };
        }
    };
    doubleIntegratorDyn()
        : dyn_pos(pos()), dyn_vel(vel()), vel_zero_constr(zero_vel()) {
        std::tie(r, r_next) = make_state("pos", 3);
        std::tie(v, v_next) = make_state("vel", 3);
        a = make_input("acc", 3);
        dyn_pos->add_arguments({r, r_next, v_next});
        dyn_vel->add_arguments({v, v_next, a});
        vel_zero_constr->add_arguments({v_next});
        // constr trial("trial", 3, __eq_cstr_s);
        // trial->add_argument(r);
        // trial->value = [=](auto &d) { d.v_ = d(r); };
        // trial->jacobian = [](sparse_approx_data &d) { d.jac_[0].setIdentity(); };
        add({dyn_pos, dyn_vel});
    }
};
// another way
// class doubleIntegrator : dynamics, public constr
// or just implement a method returning a func::collection
} // namespace atri

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP