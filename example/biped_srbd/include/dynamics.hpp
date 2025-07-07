#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <moto/core/expr.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/dynamics.hpp>

namespace biped_srbd {
using namespace moto;
struct srbd_dynamics : public dynamics {
    sym r, r_n;  // position
    sym v, v_n;  // velocity
    sym f_l;     // left foot force
    sym f_r;     // right foot force
    scalar_t m;  // mass
    scalar_t dt; // time step
    srbd_dynamics() {
        std::tie(r, r_n) = make_state("r", 3);
        std::tie(v, v_n) = make_state("v", 3);
        f_l = make_input("f_l", 3);
        f_r = make_input("f_r", 3);
        m = 1.0;
        dt = 0.01;
    }

    auto euler() {
        return constr("euler_pos", {r_n, r, v_n}, r_n - (r + v_n * dt), approx_order::first, __dyn);
    }
};

} // namespace biped_srbd

#endif // DYNAMICS_HPP