#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <moto/ocp/cost.hpp>

namespace moto {

/**
 * @brief An example of how to write a neat sparse cost
 *
 */
struct doubleIntegratorCosts {
    struct state_cost : public impl::cost {
        vector d_r, d_v;
        state_cost(sym r, sym v) : impl::cost("dI_state_cost") {
            d_r.resize(3);
            d_r.setConstant(10);
            d_v.resize(3);
            d_v.setConstant(0.1);

            add_arguments({r, v});
            value = [&](sp_approx_map &data) {
                data.v_.noalias() += 0.5 * d_r.transpose() * data[0].cwiseAbs2();
                data.v_.noalias() += 0.5 * d_v.transpose() * data[1].cwiseAbs2();
            };
            jacobian = [this](sp_approx_map &data) { // make sure use +=
                data.jac_[0].noalias() += data[0].transpose() * d_r.asDiagonal();
                data.jac_[1].noalias() += data[1].transpose() * d_v.asDiagonal();
            };
            hessian = [this](sp_approx_map &data) {
                data.hess_[0][0].diagonal() += d_r;
                data.hess_[1][1].diagonal() += d_v;
            };
        }
    };
    struct input_cost : public impl::cost {
        vector d_a;
        input_cost(sym a) : impl::cost("dI_input_cost") {
            d_a.resize(3);
            d_a.setConstant(1e-3);
            add_arguments({a});
            value = [&](sp_approx_map &data) {
                data.v_.noalias() += 0.5 * d_a.transpose() * data[0].cwiseAbs2();
            };
            jacobian = [this](sp_approx_map &data) { // make sure use +=
                data.jac_[0].noalias() += data[0].transpose() * d_a.asDiagonal();
            };
            hessian = [this](sp_approx_map &data) {
                data.hess_[0][0].diagonal() += d_a;
            };
        }
    };
    expr_list running(sym r, sym v, sym a) {
        return {new state_cost(r, v), new input_cost(a)};
    }
    expr_list terminal(sym r, sym v) {
        return {cost(new state_cost(r, v)).as_terminal()};
    }
};

} // namespace moto

#endif // DOUBLE_INTEGRATOR_COST_HPP