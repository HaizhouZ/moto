#ifndef MOTO_ARM_COST_HPP
#define MOTO_ARM_COST_HPP

#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>

namespace moto {
struct armCosts {
    struct ee_cost : public cost {
        vector d_r;
        // ee_cost(cost &&rhs) : cost(std::move(rhs)) {};
        using cost::cost;
        inline static auto r_des = sym::create("r_des", 3, __p);
        inline static auto W_kin = sym::create("W_kin", 1, __p);
        static ee_cost *create(sym *q) {
            auto c = cost::create("kin_cost")->cast<ee_cost>();
            c->d_r.resize(3);
            c->d_r.setConstant(100);

            c->add_arguments({q, r_des, W_kin});
            c->load_external();
            return c;
        }
    };
    struct state_cost : public cost {
        vector d_q, d_v;
        static auto *create(sym *q, sym *v) {
            auto c = cost::create("dI_state_cost")->cast<state_cost>();
            c->setup(q, v);
            return c;
        }
        state_cost(expr &&rhs) : cost(std::move(rhs)) {};
        void setup(sym *q, sym *v) {
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
    struct input_cost : public cost {
        vector d_a;
        static auto create(sym *a) {
            auto c = cost::create("dI_input_cost")->cast<input_cost>();
            c->setup(a);
            return c;
        }
        input_cost(expr &&rhs) : cost(std::move(rhs)) {}
        void setup(sym *a) {
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
    static expr_list running(sym *q, sym *v, sym *a) {
        expr_list b({input_cost::create(a), input_cost::create(a), input_cost::create(a)});
        return {state_cost::create(q, v), ee_cost::create(q)};
    }
    static expr_list terminal(sym *q, sym *v) {
        return {state_cost::create(q, v)->as_terminal(), ee_cost::create(q)->as_terminal()};
    }
};

} // namespace moto

#endif // ARM_COST_HPP