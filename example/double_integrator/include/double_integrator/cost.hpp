#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <atri/ocp/func/cost.hpp>

namespace atri {

/**
 * @brief An example of how to write a neat sparse cost
 *
 */
struct doubleIntegratorCosts {
    struct state_cost : public cost {
        vector d_r, d_v;
        state_cost(sym_ptr_t r, sym_ptr_t v) : cost("dI_state_cost") {
            d_r.resize(3);
            d_r.setConstant(10);
            d_v.resize(3);
            d_v.setConstant(0.1);

            add_arguments({r, v});
        }
        void value(sparse_approx_data_ptr_t data) override {
            data->v_.noalias() += d_r.transpose() * data->in_args_[0].cwiseAbs2();
            data->v_.noalias() += d_v.transpose() * data->in_args_[1].cwiseAbs2();
        }
        void jacobian(sparse_approx_data_ptr_t data) override { // make sure use +=
            data->jac_[0].noalias() += data->in_args_[0].transpose() * d_r.asDiagonal();
            data->jac_[1].noalias() += data->in_args_[1].transpose() * d_v.asDiagonal();
        }
        void hessian(sparse_approx_data_ptr_t data) override {
            data->hess_[0][0].diagonal() += d_r;
            data->hess_[1][1].diagonal() += d_v;
        }
    };
    struct input_cost : public cost {
        vector d_a;
        input_cost(sym_ptr_t a) : cost("dI_input_cost") {
            d_a.resize(3);
            d_a.setConstant(1e-3);

            add_arguments({a});
        }
        void value(sparse_approx_data_ptr_t data) override {
            data->v_.noalias() += d_a.transpose() * data->in_args_[0].cwiseAbs2();
        }
        void jacobian(sparse_approx_data_ptr_t data) override { // make sure use +=
            data->jac_[0].noalias() += data->in_args_[0].transpose() * d_a.asDiagonal();
        }
        void hessian(sparse_approx_data_ptr_t data) override {
            data->hess_[0][0].diagonal() += d_a;
        }
    };
    collection running(sym_ptr_t r, sym_ptr_t v, sym_ptr_t a) {
        return {new state_cost(r, v), new input_cost(a)};
    }
    collection terminal(sym_ptr_t r, sym_ptr_t v) {
        return {new state_cost(r, v)};
    }
};

} // namespace atri

#endif // DOUBLE_INTEGRATOR_COST_HPP