#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <moto/core/expr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>

namespace biped_srbd {
using namespace moto;
struct srbd_dynamics {
    var r, r_n;                     // position
    var v, v_n;                     // velocity
    var r_l, r_l_n;                 // left foot position
    var r_r, r_r_n;                 // right foot position
    var v_l, v_r;                   // left/right foot velocity
    var f_l;                        // left foot force
    var f_r;                        // right foot force
    scalar_t m;                     // mass
    scalar_t dt;                    // time step
    var r_d;                        // desired position
    var active_l, active_r;         // left/right foot active
    var active_l_cur, active_r_cur; // left/right foot active
    srbd_dynamics() {
        std::tie(r, r_n) = sym::states("r", 3);
        std::tie(r_l, r_l_n) = sym::states("r_l", 3);
        std::tie(r_r, r_r_n) = sym::states("r_r", 3);
        std::tie(v, v_n) = sym::states("v", 3);
        v_l = sym::inputs("v_l", 3);
        v_r = sym::inputs("v_r", 3);
        r_d = sym::params("r_d", 3);
        active_l = sym::params("active_l", 1);
        active_r = sym::params("active_r", 1);
        active_l_cur = sym::params("active_l_cur", 1);
        active_r_cur = sym::params("active_r_cur", 1);
        f_l = sym::inputs("f_l", 3);
        f_r = sym::inputs("f_r", 3);
        m = 1.0;
        dt = 0.01;
    }

    expr_list euler() {
        return {constr("euler_pos", {r_n, r, v_n}, r_n - (r + v_n * dt), approx_order::first, __dyn),
                constr("euler_pos_l", {r_l_n, r_l, v_l}, r_l_n - (r_l + v_l * dt), approx_order::first, __dyn),
                constr("euler_pos_r", {r_r_n, r_r, v_r}, r_r_n - (r_r + v_r * dt), approx_order::first, __dyn),
                constr("euler_vel", {v_n, v, f_l, f_r}, m * (v_n - v) - (f_l + f_r + cs::SX({0, 0, -9.81})) * dt, approx_order::first, __dyn)};
    }
    expr_list friction_cone(scalar_t mu = 0.5) {
        auto make_fric_cone = [mu](const sym &f) {
            return cs::SX::vertcat({f(0) - mu * f(2),
                                    f(1) - mu * f(2),
                                    -f(0) - mu * f(2),
                                    -f(1) - mu * f(2)});
        };
        return {constr("fric_l", {f_l}, make_fric_cone(f_l)).as_ineq<ipm>(),
                constr("fric_r", {f_r}, make_fric_cone(f_r)).as_ineq<ipm>()};
    }
    expr_list foot_loc_constr() {
        return {constr("foot_overlap", {r_l, r_r}, cs::SX::sumsqr(r_l - r_r) - 0.16).as_ineq<ipm>()}; // 4cm foot
    }
    expr_list stance_foot_constr() {
        return {constr("stance_u",
                       {v_l, f_l, active_l_cur, v_r, f_r, active_r_cur},
                       cs::SX::vertcat({active_l_cur * v_l(cs::Slice(0, 2)), (1 - active_l_cur) * f_l,
                                        active_r_cur * v_r(cs::Slice(0, 2)), (1 - active_r_cur) * f_r}))
                    .as_eq(),
                constr("stance_z",
                       {r_l_n, r_r_n, active_l, active_r},
                       cs::SX::vertcat({active_l * r_l_n(2), active_r * r_r_n(2)}))
                    .as_eq()};
    }
    expr_list running_cost() {
        auto running_cost = 100 * cs::SX::sumsqr(r - r_d) +                      // position cost
                            cs::SX::sumsqr(v) +                                  // velocity cost
                            1e-3 * (cs::SX::sumsqr(f_l) + cs::SX::sumsqr(f_r)) + // force cost
                            0.1 * (cs::SX::sumsqr(v_l) + cs::SX::sumsqr(v_r));   // foot velocity cost
        return {cost("srbd_cost", {r, v, r_l, r_r, f_l, f_r, v_l, v_r, r_d}, dt * running_cost)};
    }
    expr_list terminal_cost() {
        auto running_cost = 100 * cs::SX::sumsqr(r - r_d) +          // position cost
                            cs::SX::sumsqr(v) +                      // velocity cost
                            cs::SX::sumsqr((r_l))(cs::Slice(0, 2)) + // foot position cost
                            cs::SX::sumsqr((r_r))(cs::Slice(0, 2));  // foot position cost
        return {cost("srbd_cost", {r, v, r_l, r_r, r_d}, running_cost).as_terminal()};
    }
};

} // namespace biped_srbd

#endif // DYNAMICS_HPP