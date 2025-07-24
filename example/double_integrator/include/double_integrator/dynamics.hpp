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
    auto pos() {
        auto d = constr("doubleIntegratorDynamics_pos", approx_order::first, 3, __dyn);
        d.value() = [*this](func_approx_map &data) {
            data.v_ = -data[r] + data[r_next] - 0.01 * data[v_next];
        };
        d.jacobian() = [*this](func_approx_map &data) {
            data.jac(r).diagonal().setConstant(-1);
            data.jac(r_next).setIdentity();
            data.jac(v_next).diagonal().setConstant(-0.01);
        };
        d.add_arguments({r, r_next, v_next});
        return d;
    }
    auto vel() {
        auto d = constr("doubleIntegratorDynamics_vel", approx_order::first, 3, __dyn);
        d.value() = [this](func_approx_map &data) {
            data.v_ = -data[v] + data[v_next] - 0.01 * data[a];
        };
        d.jacobian() = [this](func_approx_map &data) {
            data.jac(v).diagonal().setConstant(-1);
            data.jac(v_next).setIdentity();
            data.jac(a).diagonal().setConstant(-0.01);
        };
        d.add_arguments({v, v_next, a});
        return d;
    }
    auto zero_vel() {
        auto d = constr("doubleIntegratorDynamics_zero_vel", approx_order::first, 3, __eq_x);
        d.value() = [this](func_approx_map &data) {
            data.v_ = data[v];
        };
        d.jacobian() = [this](func_approx_map &data) {
            data.jac(v).setIdentity();
        };
        d.add_arguments({v});
        return d;
    }
    doubleIntegratorDyn() {
        std::tie(r, r_next) = sym::states("pos", 3);
        std::tie(v, v_next) = sym::states("vel", 3);
        a = sym::inputs("acc", 3);
        dyn_pos = pos();
        dyn_vel = vel();
        vel_zero_constr = zero_vel();
        assign({dyn_pos, dyn_vel});
    }
};
// another way
// class doubleIntegrator : dynamics, public constr
// or just implement a method returning a func::expr_list
} // namespace moto

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP