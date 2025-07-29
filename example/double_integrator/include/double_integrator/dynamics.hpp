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
    var r, v, a, r_next, v_next;
    func dyn_pos, dyn_vel, vel_zero_constr;

    doubleIntegratorDyn() {
        std::tie(r, r_next) = sym::states("pos", 3);
        std::tie(v, v_next) = sym::states("vel", 3);
        a = sym::inputs("acc", 3);
        dyn_pos = make_dyn_pos();
        dyn_vel = make_dyn_vel();
        vel_zero_constr = make_zero_vel();
        assign({dyn_pos, dyn_vel});
    }

private:
    func make_dyn_pos() {
        auto d = constr("doubleIntegratorDynamics_pos", approx_order::first, 3, __dyn);
        d->value = [](func_approx_data &data) {
            data.v_ = -data[0] + data[1] - 0.01 * data[2];
        };
        d->jacobian = [this](func_approx_data &data) {
            data.jac(0).diagonal().setConstant(-1);
            data.jac(1).setIdentity();
            data.jac(2).diagonal().setConstant(-0.01);
        };
        d->add_arguments({r, r_next, v_next});
        return d;
    }

    func make_dyn_vel() {
        auto d = constr("doubleIntegratorDynamics_vel", approx_order::first, 3, __dyn);
        d->value = [](func_approx_data &data) {
            data.v_ = -data[0] + data[1] - 0.01 * data[2];
        };
        d->jacobian = [](func_approx_data &data) {
            data.jac(0).diagonal().setConstant(-1);
            data.jac(1).setIdentity();
            data.jac(2).diagonal().setConstant(-0.01);
        };
        d->add_arguments({v, v_next, a});
        return d;
    }

    func make_zero_vel() {
        auto d = constr("doubleIntegratorDynamics_zero_vel", approx_order::first, 3, __eq_x);
        d->value = [](func_approx_data &data) {
            data.v_ = data[0];
        };
        d->jacobian = [](func_approx_data &data) {
            data.jac(0).setIdentity();
        };
        d->add_arguments({v});
        return d;
    }
};

} // namespace moto

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP