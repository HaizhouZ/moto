#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#include <moto/ocp/constr.hpp>

namespace moto {

/**
 * @brief double integrator dynamics
 * v_next - v - a * dt = 0
 * r_next - r - v_next * dt = 0
 */
class doubleIntegratorDyn : public expr_list {
  public:
    // position, velocity, acceleration, position, velocity
    sym r, v, a, r_next, v_next;
    constr dyn_pos, dyn_vel, vel_zero_constr;
    struct pos : public constr {
        pos() : constr("doubleIntegratorDynamics_pos", approx_order::first, 3, __dyn) {
            value() = [](func_approx_map &data) {
                data.v_ = -data[0] + data[1] - 0.01 * data[2];
            };
            jacobian() = [](func_approx_map &data) {
                data.jac_[0].diagonal().setConstant(-1);
                data.jac_[1].setIdentity();
                data.jac_[2].diagonal().setConstant(-0.01);
            };
        }
    };
    struct vel : public constr {
        vel() : constr("doubleIntegratorDynamics_vel", approx_order::first, 3, __dyn) {
            value() = [](func_approx_map &data) {
                data.v_ = -data[0] + data[1] - 0.01 * data[2];
            };
            jacobian() = [](func_approx_map &data) {
                data.jac_[0].diagonal().setConstant(-1);
                data.jac_[1].setIdentity();
                data.jac_[2].diagonal().setConstant(-0.01);
            };
        }
    };
    struct zero_vel : public constr {
        zero_vel() : constr("doubleIntegratorDynamics_zero_vel", approx_order::first, 3, __eq_x) {
            value() = [](func_approx_map &data) {
                data.v_ = data[0];
            };
            jacobian() = [](func_approx_map &data) {
                data.jac_[0].setIdentity();
            };
        }
    };
    doubleIntegratorDyn()
        : dyn_pos(pos()), dyn_vel(vel()), vel_zero_constr(zero_vel()) {
        std::tie(r, r_next) = sym::states("pos", 3);
        std::tie(v, v_next) = sym::states("vel", 3);
        a = sym::inputs("acc", 3);
        dyn_pos.add_arguments({r, r_next, v_next});
        dyn_vel.add_arguments({v, v_next, a});
        vel_zero_constr.add_arguments({v});
        // constr trial("trial", 3, __eq_x);
        // trial->add_argument(r);
        // trial->value = [=](auto &d) { d.v_ = d(r); };
        // trial->jacobian = [](func_approx_map &d) { d.jac_[0].setIdentity(); };
        assign({dyn_pos, dyn_vel});
    }
};
// another way
// class doubleIntegrator : dynamics, public constr
// or just implement a method returning a func::expr_list
} // namespace moto

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP