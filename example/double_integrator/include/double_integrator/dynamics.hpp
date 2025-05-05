#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#include <atri/ocp/constr.hpp>

namespace atri {
/**
 * @brief expression collection trait
 * expr_ can only be accessed via get() which requires an setup() must be called
 */
struct expr_collection {
  private:
    std::vector<expr_ptr_t> expr_;

  protected:
    void add(std::initializer_list<expr_ptr_t> exprs) { expr_.insert(expr_.end(), exprs); }

  public:
    auto get_expr() {
        assert(!expr_.empty());
        return expr_;
    }
};

// trait
struct dynamics {
  protected:
    static auto make_input(const std::string &name, size_t dim) {
        return std::make_shared<sym>(name, dim, __u);
    }
    static auto make_state(const std::string &name, size_t dim) {
        auto temp = std::make_shared<sym>(name, dim, __x);
        auto next = std::make_shared<sym>(name + "_nxt", dim, __y);
        return std::make_pair(temp, next);
    }
};
/**
 * @brief double integrator dynamics
 * v_next - v - a * dt = 0
 * r_next - r - v_next * dt = 0
 */
class doubleIntegratorDyn : dynamics, public expr_collection {
  public:
    // position, velocity, acceleration, position, velocity
    sym_ptr_t r, v, a, r_next, v_next;
    constr_ptr_t dyn_pos, dyn_vel;
    struct pos : public constr {
        pos() : constr("doubleIntegratorDynamics_pos", 3, __dyn, approx_order::first) {}
        void jacobian(sparse_approx_data_ptr_t data) override {
            data->jac_[0].diagonal().setConstant(-1);
            data->jac_[1].setIdentity();
            data->jac_[2].diagonal().setConstant(-0.01);
        };
    };
    struct vel : public constr {
        vel() : constr("doubleIntegratorDynamics_vel", 3, __dyn, approx_order::first) {}
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
        expr_collection::add({r, v, a, r_next, v_next, dyn_pos, dyn_vel});
    }
};
// another way
// class doubleIntegrator : dynamics, public constr
} // namespace atri

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP