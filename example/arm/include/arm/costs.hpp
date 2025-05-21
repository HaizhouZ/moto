#ifndef ATRI_ARM_COST_HPP
#define ATRI_ARM_COST_HPP

#include <atri/core/external_function.hpp>
#include <atri/ocp/cost.hpp>

namespace atri {

struct armCosts {
    struct ee_cost : public cost_impl {
        vector d_r;
        inline static sym r_des{"r_des", 3, __p};
        ee_cost(sym q) : cost_impl("ee_cost") {
            d_r.resize(3);
            d_r.setConstant(100);

            add_arguments({q, r_des});
            value = [](sparse_approx_data &data) {
                static ext_func kin_cost("gen/libkin_cost.so", "kin_cost");
                kin_cost.invoke(data.in_args_, data.v_);
            };
            jacobian = [this](sparse_approx_data &data) { // make sure use +=
                static ext_func kin_cost("gen/libkin_cost_jac.so", "kin_cost_jac");
                kin_cost.invoke(data.in_args_, data.jac_[0]);
            };
            hessian = [this](sparse_approx_data &data) {
                static ext_func kin_cost("gen/libkin_cost_hess.so", "kin_cost_hess");
                /// @todo: automated this
                kin_cost.invoke(data.in_args_, data.hess_[0][0]);
            };
        }
    };
    struct state_cost : public cost_impl {
        vector d_q, d_v;
        state_cost(sym q, sym v) : cost_impl("dI_state_cost") {
            d_q.resize(7);
            d_q.setConstant(10);
            d_v.resize(7);
            d_v.setConstant(0.1);

            add_arguments({q, v});
            value = [&](sparse_approx_data &data) {
                data.v_.noalias() += 0.5 * d_q.transpose() * data.in_args_[0].cwiseAbs2();
                data.v_.noalias() += 0.5 * d_v.transpose() * data.in_args_[1].cwiseAbs2();
            };
            jacobian = [this](sparse_approx_data &data) { // make sure use +=
                data.jac_[0].noalias() += data.in_args_[0].transpose() * d_q.asDiagonal();
                data.jac_[1].noalias() += data.in_args_[1].transpose() * d_v.asDiagonal();
            };
            hessian = [this](sparse_approx_data &data) {
                data.hess_[0][0].diagonal() += d_q;
                data.hess_[1][1].diagonal() += d_v;
            };
        }
    };
    struct input_cost : public cost_impl {
        vector d_a;
        input_cost(sym a) : cost_impl("dI_input_cost") {
            d_a.resize(7);
            d_a.setConstant(1e-3);
            add_arguments({a});
            value = [&](sparse_approx_data &data) {
                data.v_.noalias() += 0.5 * d_a.transpose() * data.in_args_[0].cwiseAbs2();
            };
            jacobian = [this](sparse_approx_data &data) { // make sure use +=
                data.jac_[0].noalias() += data.in_args_[0].transpose() * d_a.asDiagonal();
            };
            hessian = [this](sparse_approx_data &data) {
                data.hess_[0][0].diagonal() += d_a;
            };
        }
    };
    static collection running(sym q, sym v, sym a) {
        return {new state_cost(q, v), new input_cost(a)};//, new ee_cost(q)};
    }
    static collection terminal(sym q, sym v) {
        return {new state_cost(q, v)};//, new ee_cost(q)};
    }
};

} // namespace atri

#endif // ARM_COST_HPP