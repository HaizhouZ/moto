#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <atri/ocp/cost.hpp>

namespace atri {

/**
 * @brief simple quadratic cost
 *
 */
struct doubleIntegratorCost : public cost {
    vector d_r, d_v, d_a;
    doubleIntegratorCost(sym_ptr_t r, sym_ptr_t v, sym_ptr_t a) : cost("doubleIntegratorCost") {
        d_r.resize(3);
        d_r.setConstant(10);
        d_v.resize(3);
        d_v.setConstant(0.1);
        d_a.resize(3);
        d_a.setConstant(1e-3);

        add_arguments({r, v, a});
    }
    void jacobian_impl(sparse_approx_data_ptr_t data) override { // make sure use +=
        data->jac_[0].noalias() += data->in_args_[0].transpose() * d_r.asDiagonal();
        data->jac_[1].noalias() += data->in_args_[1].transpose() * d_v.asDiagonal();
        data->jac_[2].noalias() += data->in_args_[2].transpose() * d_a.asDiagonal();
    }
    void hessian_impl(sparse_approx_data_ptr_t data) override {
        data->hess_[0][0].diagonal() += d_r;
        data->hess_[1][1].diagonal() += d_v;
        data->hess_[2][2].diagonal() += d_a;
    }
};

} // namespace atri

#endif // DOUBLE_INTEGRATOR_COST_HPP