#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#include <atri/ocp/func/constr.hpp>
#include <atri/ocp/func/dynamics.hpp>

namespace atri {

/**
 * @brief double integrator dynamics
 * v_next - v - a * dt = 0
 * r_next - r - v_next * dt = 0
 */
class doubleIntegratorDyn : public dynamics, public collection {
  public:
    // position, velocity, acceleration, position, velocity
    sym_ptr_t r, v, a, r_next, v_next;
    constr_ptr_t dyn_pos, dyn_vel;
    struct pos : public constr {
        pos() : constr("doubleIntegratorDynamics_pos", 3, __dyn, approx_order::first) {}
        void value(sparse_approx_data_ptr_t data) override {
            data->v_ = -data->in_args_[0] + data->in_args_[2] - 0.01 * data->in_args_[1];
        };
        void jacobian(sparse_approx_data_ptr_t data) override {
            data->jac_[0].diagonal().setConstant(-1);
            data->jac_[1].setIdentity();
            data->jac_[2].diagonal().setConstant(-0.01);
        };
    };
    struct vel : public constr {
        vel() : constr("doubleIntegratorDynamics_vel", 3, __dyn, approx_order::first) {}
        void value(sparse_approx_data_ptr_t data) override {
            data->v_ = -data->in_args_[0] + data->in_args_[2] - 0.01 * data->in_args_[1];
        };
        void jacobian(sparse_approx_data_ptr_t data) override {
            data->jac_[0].diagonal().setConstant(-1);
            data->jac_[1].setIdentity();
            data->jac_[2].diagonal().setConstant(-0.01);
        };
    };
    doubleIntegratorDyn()
        : dyn_pos(new pos()),
          dyn_vel(new vel()) {
        std::tie(r, r_next) = make_state("pos", 3);
        std::tie(v, v_next) = make_state("vel", 3);
        a = make_input("acc", 3);
        dyn_pos->add_arguments({r, r_next, v_next});
        dyn_vel->add_arguments({v, v_next, a});
        add({r, v, a, r_next, v_next, dyn_pos, dyn_vel});
    }
};
// another way
// class doubleIntegrator : dynamics, public constr
// or just implement a method returning a func::collection
} // namespace atri

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP