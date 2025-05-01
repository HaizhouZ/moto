#ifndef DOUBLE_INTEGRATOR_DYNAMICS_HPP
#define DOUBLE_INTEGRATOR_DYNAMICS_HPP

#include <atri/ocp/constr.hpp>

namespace atri {

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

class doubleIntegratorDyn : dynamics, public constr {
  public:
    // position, velocity, acceleration, position, velocity
    sym_ptr_t r, v, a, r_next, v_next;
    doubleIntegratorDyn()
        : constr("doubleIntegratorDynamics", 3, __dyn, approx_order::first) {
        std::tie(r, r_next) = make_state("pos", 3);
        std::tie(v, v_next) = make_state("vel", 3);
        a = make_input("acc", 3);
        add_arguments({r, v, a, r_next, v_next});
    }

  private:
    // Add private members or helper functions if needed
};



} // namespace atri

#endif // DOUBLE_INTEGRATOR_DYNAMICS_HPP