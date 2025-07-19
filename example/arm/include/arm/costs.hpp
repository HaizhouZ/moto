#ifndef MOTO_ARM_COST_HPP
#define MOTO_ARM_COST_HPP

#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
struct armCosts {
    struct ee_cost : public impl::cost {
        vector d_r;
        inline static sym r_des{"r_des", 3, __p};
        inline static sym W_kin{"W_kin", 1, __p};
        ee_cost(sym q) : impl::cost("kin_cost") {
            d_r.resize(3);
            d_r.setConstant(100);

            add_arguments({q, r_des, W_kin});
            load_external();
        }
    };
    struct state_cost : public impl::cost {
        vector d_q, d_v;
        state_cost(sym q, sym v) : impl::cost("dI_state_cost") {
            d_q.resize(7);
            d_q.setConstant(10);
            d_v.resize(7);
            d_v.setConstant(0.1);

            add_arguments({q, v});
            value = [&](func_approx_map &data) {
                data.v_.noalias() += 0.5 * d_q.transpose() * data[0].cwiseAbs2();
                data.v_.noalias() += 0.5 * d_v.transpose() * data[1].cwiseAbs2();
            };
            jacobian = [this](func_approx_map &data) { // make sure use +=
                data.jac_[0].noalias() += data[0].transpose() * d_q.asDiagonal();
                data.jac_[1].noalias() += data[1].transpose() * d_v.asDiagonal();
            };
            hessian = [this](func_approx_map &data) {
                data.hess_[0][0].diagonal() += d_q;
                data.hess_[1][1].diagonal() += d_v;
            };
        }
    };
    struct input_cost : public impl::cost {
        vector d_a;
        input_cost(sym a) : impl::cost("dI_input_cost") {
            d_a.resize(7);
            d_a.setConstant(1e-2);
            add_arguments({a});
            value = [&](func_approx_map &data) {
                data.v_.noalias() += 0.5 * d_a.transpose() * data[0].cwiseAbs2();
            };
            jacobian = [this](func_approx_map &data) { // make sure use +=
                data.jac_[0].noalias() += data[0].transpose() * d_a.asDiagonal();
            };
            hessian = [this](func_approx_map &data) {
                data.hess_[0][0].diagonal() += d_a;
            };
        }
    };
    static expr_list running(sym q, sym v, sym a) {
        return {new state_cost(q, v), new input_cost(a), new ee_cost(q)};
    }
    static expr_list terminal(sym q, sym v) {
        return {cost(new state_cost(q, v)).as_terminal(), cost(new ee_cost(q)).as_terminal()};
    }
};

} // namespace moto

#endif // ARM_COST_HPP