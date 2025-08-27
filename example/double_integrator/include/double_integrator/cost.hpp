#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <moto/ocp/cost.hpp>

namespace moto {

/**
 * @brief An example of how to write a neat sparse cost
 *
 */
struct doubleIntegratorCosts {
    vector d_r = vector::Constant(3, 10);
    vector d_v = vector::Constant(3, 0.1); ///< derivatives of the cost wrt r and v
    cost state_cost(var &r, var &v) {
        auto c = cost("dI_state_cost");
        c->value = [=, this](func_approx_data &data) {
            data.v_.noalias() += 0.5 * d_r.transpose() * data[r].cwiseAbs2();
            data.v_.noalias() += 0.5 * d_v.transpose() * data[v].cwiseAbs2();
        };
        c->jacobian = [=, this](func_approx_data &data) { // make sure use +=
            data.jac_[0].noalias() += data[0].transpose() * d_r.asDiagonal();
            data.jac_[1].noalias() += data[1].transpose() * d_v.asDiagonal();
        };
        c->hessian = [=, this](func_approx_data &data) {
            // data.merit_hess_[0][0].diagonal() += d_r;
            // data.merit_hess_[1][1].diagonal() += d_v;
            data.merit_hess_[0][0] += d_r;
            data.merit_hess_[1][1] += d_v;
        };
        c->add_arguments({r, v});
        c.set_diag_hess();
        return c;
    }

    vector d_a = vector::Constant(3, 1e-3);
    cost input_cost(var &a) {
        auto c = cost("dI_input_cost");
        c->value = [=, this](func_approx_data &data) {
            data.v_.noalias() += 0.5 * d_a.transpose() * data[a].cwiseAbs2();
        };
        c->jacobian = [=, this](func_approx_data &data) {
            data.jac_[0].noalias() += data[0].transpose() * d_a.asDiagonal();
        };
        c->hessian = [=, this](func_approx_data &data) {
            // data.merit_hess_[0][0].diagonal() += d_a;
            data.merit_hess_[0][0] += d_a;
        };
        c->add_arguments({a});
        c.set_diag_hess();
        return c;
    }

    expr_list running(var &r, var &v, var &a) {
        return {state_cost(r, v), input_cost(a)};
    }
    expr_list terminal(var &r, var &v) {
        return {state_cost(r, v).as_terminal()};
    }
};

} // namespace moto

#endif // DOUBLE_INTEGRATOR_COST_HPP