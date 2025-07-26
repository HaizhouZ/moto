#ifndef DOUBLE_INTEGRATOR_COST_HPP
#define DOUBLE_INTEGRATOR_COST_HPP

#include <moto/ocp/cost.hpp>

namespace moto {

/**
 * @brief An example of how to write a neat sparse cost
 *
 */
struct doubleIntegratorCosts {
    auto state_cost(sym r, sym v) {
        struct impl : public cost::impl {
            vector d_r = vector::Constant(3, 10);
            vector d_v = vector::Constant(3, 0.1); ///< derivatives of the cost wrt r and v
            using cost::impl;                ///< inherit constructor from cost::impl
            impl(sym r, sym v, cost::impl &&rhs) : cost::impl(std::move(rhs)) {
                value = [=](func_approx_map &data) {
                    data.v_.noalias() += 0.5 * d_r.transpose() * data[r].cwiseAbs2();
                    data.v_.noalias() += 0.5 * d_v.transpose() * data[v].cwiseAbs2();
                };
                jacobian = [=](func_approx_map &data) { // make sure use +=
                    data.jac_[0].noalias() += data[0].transpose() * d_r.asDiagonal();
                    data.jac_[1].noalias() += data[1].transpose() * d_v.asDiagonal();
                };
                hessian = [=](func_approx_map &data) {
                    data.hess_[0][0].diagonal() += d_r;
                    data.hess_[1][1].diagonal() += d_v;
                };
            } ///< move constructor
            ~impl() = default; ///< destructor
        };
        cost c("dI_state_cost");
        c.add_arguments({r, v});
        c.set_impl(new impl(r, v, std::move(c.get_impl())));
        return c;
    }
    auto input_cost(sym a) {
        struct impl : public cost::impl {
            vector d_a = vector::Constant(3, 1e-3);
            using cost::impl;
            impl(sym a, cost::impl &&rhs) : cost::impl(std::move(rhs)) {
                value = [=](func_approx_map &data) {
                    data.v_.noalias() += 0.5 * d_a.transpose() * data[a].cwiseAbs2();
                };
                jacobian = [=](func_approx_map &data) {
                    data.jac_[0].noalias() += data[0].transpose() * d_a.asDiagonal();
                };
                hessian = [=](func_approx_map &data) {
                    data.hess_[0][0].diagonal() += d_a;
                };
            }
            ~impl() = default;
        };
        cost c("dI_input_cost");
        c.add_arguments({a});
        c.set_impl(new impl(a, std::move(c.get_impl())));
        return c;
    }
    expr_list running(sym r, sym v, sym a) {
        return {state_cost(r, v), input_cost(a)};
    }
    expr_list terminal(sym r, sym v) {
        return {state_cost(r, v).as_terminal()};
    }
};

} // namespace moto

#endif // DOUBLE_INTEGRATOR_COST_HPP