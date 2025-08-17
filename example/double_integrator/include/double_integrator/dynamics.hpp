#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#define DEFAULT_DENSE true
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/euler_dynamics.hpp>

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
    func dyn, vel_zero_constr;

    doubleIntegratorDyn(bool dense = DEFAULT_DENSE) {
        if (dense) {
            std::tie(r, r_next) = sym::states("pos", 3);
            std::tie(v, v_next) = sym::states("vel", 3);
            a = sym::inputs("acc", 3);
        }
        std::cout << "doubleIntegratorDyn: " << (dense ? "dense" : "sparse") << std::endl;
        dyn = make_dyn_pos(dense);
        vel_zero_constr = make_zero_vel();
        assign({dyn});
    }

  private:
    func make_dyn_pos(bool dense) {
        if (dense) {
            auto d = dense_dynamics("doubleIntegratorDynamics_pos", approx_order::first, 6);
            d->value = [this](func_approx_data &data) {
                // data.v_.head<3>() = data[r_next] - data[r] - 0.01 * data[v_next];
                // data.v_.tail<3>() = data[v_next] - data[v] - 0.01 * data[a];
                data.v_.head<3>() = data[r_next] - data[r] - 0.01 * data[v];
                data.v_.tail<3>() = data[v_next] - data[v] - 0.01 * data[a];
            };
            d->jacobian = [this](func_approx_data &data) {
                data.jac(r_next).topRows<3>().diagonal().setOnes();
                data.jac(r).topRows<3>().diagonal().setConstant(-1);
                // data.jac(v_next).topRows<3>().diagonal().setConstant(-0.01);
                data.jac(v).topRows<3>().diagonal().setConstant(-0.01);
                data.jac(v_next).bottomRows<3>().diagonal().setOnes();
                data.jac(v).bottomRows<3>().diagonal().setConstant(-1);
                data.jac(a).bottomRows<3>().diagonal().setConstant(-0.01);
            };
            d->add_arguments({r, r_next, v, v_next, a});
            return d;
        } else {
            auto d = explicit_euler("doubleIntegratorDynamics_pos");
            std::tie(r, r_next, v, v_next, a) = d.create_2nd_ord_vars("dInt", 3);
            d.add_dt(0.01);
            return d;
        }
    }

    func make_zero_vel() {
        auto d = constr("doubleIntegratorDynamics_zero_vel", approx_order::first, 3, __eq_x);
        d->value = [](func_approx_data &data) {
            data.v_ = data[0];
        };
        d->jacobian = [](func_approx_data &data) {
            data.jac(0).diagonal().setOnes();
        };
        d->add_arguments({v});
        return d;
    }
};

} // namespace moto

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP