#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/dynamics/euler_dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
namespace biped_srbd {
using namespace moto;
struct srbd_dynamics {
    var r, r_n;                     // position
    var v, v_n;                     // velocity
    var a;                          // acceleration
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
    func euler_dyn;
    bool dense;

    struct sparse_dyn : public generic_dynamics {
        var r, r_n;     // position
        var v, v_n;     // velocity
        var r_l, r_l_n; // left foot position
        var r_r, r_r_n; // right foot position
        var v_l, v_r;   // left/right foot velocity
        var f_l;        // left foot force
        var f_r;        // right foot force
        scalar_t dt;
        scalar_t m;
        sparse_dyn(scalar_t m, scalar_t dt)
            : generic_dynamics("srbd_sparse_dyn", approx_order::first, dim_tbd, __dyn), m(m), dt(dt) {
            std::tie(r, r_n) = sym::states("r", 3);
            std::tie(r_l, r_l_n) = sym::states("r_l", 3);
            std::tie(r_r, r_r_n) = sym::states("r_r", 3);
            std::tie(v, v_n) = sym::states("v", 3);
            v_l = sym::inputs("v_l", 3);
            v_r = sym::inputs("v_r", 3);
            f_l = sym::inputs("f_l", 3);
            f_r = sym::inputs("f_r", 3);
            dim_ = 12;
            set_from_casadi(var_inarg_list{r, r_n, r_l, r_l_n, r_r, r_r_n, v, v_n, v_l, v_r, f_l, f_r},
                            cs::SX::vertcat({r_n - (r + v * dt),
                                             r_l_n - (r_l + v_l * dt),
                                             r_r_n - (r_r + v_r * dt),
                                             m * (v_n - v) - (f_l + f_r + cs::SX({0, 0, -9.81})) * dt}));
        }

        struct approx_data : public generic_dynamics::approx_data {
            approx_order order;
            approx_data(base::approx_data &&rhs) : generic_dynamics::approx_data(std::move(rhs)) {
                auto &dyn = static_cast<const sparse_dyn &>(func_);
                auto prob = problem();
                auto f_st = prob->get_expr_start(func_);
                approx_->jac_[__y].insert<sparsity::diag>(f_st, prob->get_expr_start(dyn.r_n),
                                                          dyn.arg_dim(__y))
                    .setConstant(1.0)
                    .bottomRows(3)
                    .setConstant(dyn.m);
                auto x_diag = approx_->jac_[__x].insert<sparsity::diag>(f_st, prob->get_expr_start(dyn.r),
                                                                        dyn.arg_dim(__x));
                x_diag.setConstant(-1.0).bottomRows(3).setConstant(-dyn.m);
                dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, prob->get_expr_start(dyn.r),
                                                            dyn.arg_dim(__x)) = x_diag;
                approx_->jac_[__x].insert<sparsity::diag>(f_st, prob->get_expr_start(dyn.v), dyn.v->dim()).setConstant(-dyn.dt);
                dyn_proj_->proj_f_x_.insert<sparsity::diag>(f_st, prob->get_expr_start(dyn.v), dyn.v->dim()).setConstant(-dyn.dt);
                approx_->jac_[__u].insert<sparsity::diag>(f_st + 3, prob->get_expr_start(dyn.v_l), 9).setConstant(-dyn.dt);
                approx_->jac_[__u].insert<sparsity::diag>(f_st + 9, prob->get_expr_start(dyn.f_r), 3).setConstant(-dyn.dt);
                dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st + 3, prob->get_expr_start(dyn.v_l), 9).setConstant(-dyn.dt);
                dyn_proj_->proj_f_u_.insert<sparsity::diag>(f_st + 9, prob->get_expr_start(dyn.f_r), 3).setConstant(-dyn.dt);
                // fmt::print("sparse dyn jac x: \n{}\n", approx_->jac_[__x].dense());
                // fmt::print("sparse dyn jac u: \n{}\n", approx_->jac_[__u].dense());
                // fmt::print("sparse dyn jac y: \n{}\n", approx_->jac_[__y].dense());
                // throw std::runtime_error("sparse dyn not implemented");
            }
        };
        void jacobian_impl(func_approx_data &data) const override {};
        void compute_project_derivatives(func_approx_data &data) const override {
            auto &d = data.as<approx_data>();
            d.proj_f_res_ = d.v_;
        }
        func_approx_data_ptr_t create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const override {
            return func_approx_data_ptr_t(make_approx<sparse_dyn>(primal, raw, shared));
        }
    };

    srbd_dynamics(bool dense = true) : dense(dense) {
        m = 1.0;
        dt = 0.01;
        if (dense) {
            std::tie(r, r_n) = sym::states("r", 3);
            std::tie(r_l, r_l_n) = sym::states("r_l", 3);
            std::tie(r_r, r_r_n) = sym::states("r_r", 3);
            std::tie(v, v_n) = sym::states("v", 3);
            v_l = sym::inputs("v_l", 3);
            v_r = sym::inputs("v_r", 3);
            f_l = sym::inputs("f_l", 3);
            f_r = sym::inputs("f_r", 3);
        } else {
            // auto dyn = moto::euler("srbd_euler");
            // std::tie(r, r_n, v, v_n, a) = dyn.create_2nd_ord_lin("r", 3, false);
            // std::tie(r_l, r_l_n, v_l) = dyn.create_1st_ord_lin("r_l", 3);
            // std::tie(r_r, r_r_n, v_r) = dyn.create_1st_ord_lin("r_r", 3);
            // dyn.add_dt(dt);
            auto dyn = sparse_dyn(m, dt);
            std::tie(r, r_n) = std::make_tuple(dyn.r, dyn.r_n);
            std::tie(r_l, r_l_n) = std::make_tuple(dyn.r_l, dyn.r_l_n);
            std::tie(r_r, r_r_n) = std::make_tuple(dyn.r_r, dyn.r_r_n);
            std::tie(v, v_n) = std::make_tuple(dyn.v, dyn.v_n);
            std::tie(v_l, v_r) = std::make_tuple(dyn.v_l, dyn.v_r);
            std::tie(f_l, f_r) = std::make_tuple(dyn.f_l, dyn.f_r);
            euler_dyn = func(std::move(dyn));
        }
        r_d = sym::params("r_d", 3);
        active_l = sym::params("active_l", 1);
        active_r = sym::params("active_r", 1);
        active_l_cur = sym::params("active_l_cur", 1);
        active_r_cur = sym::params("active_r_cur", 1);
    }

    expr_list euler() {
        if (dense) {
            auto args = var_inarg_list{r, r_n, v, v_n, r_l, r_l_n, r_r, r_r_n, v_l, v_r, f_l, f_r};
            auto out = cs::SX::vertcat({r_n - (r + v_n * dt),
                                        r_l_n - (r_l + v_l * dt),
                                        r_r_n - (r_r + v_r * dt),
                                        m * (v_n - v) - (f_l + f_r + cs::SX({0, 0, -9.81})) * dt});
            return {dense_dynamics("srbd_euler", args, out, approx_order::first)};
        } else {
            // return {euler_dyn, constr("inv_dyn", {a, f_l, f_r}, m * a - (f_l + f_r + cs::SX({0, 0, -9.81})))};
            return {euler_dyn};
        }
    }
    expr_list friction_cone(scalar_t mu = 0.5) {
        auto make_fric_cone = [mu](const sym &f) {
            return cs::SX::vertcat({f(0) - mu * f(2),
                                    f(1) - mu * f(2),
                                    -f(0) - mu * f(2),
                                    -f(1) - mu * f(2)});
        };
        return {constr("fric_r", {f_r}, make_fric_cone(f_r)).as_ineq<ipm>(), constr("fric_l", {f_l}, make_fric_cone(f_l)).as_ineq<ipm>()};
    }
    expr_list foot_loc_constr() {
        return {constr("foot_overlap", {r_l, r_r}, cs::SX::sumsqr(r_l - r_r) - 0.16).as_ineq<ipm>()}; // 4cm foot
    }
    expr_list stance_foot_constr() {
        return {constr("stance_u",
                       {v_l, f_l, active_l_cur, v_r, f_r, active_r_cur},
                       cs::SX::vertcat({active_l_cur * v_l(cs::Slice(0, 2)), (1 - active_l_cur) * f_l,
                                        active_r_cur * v_r(cs::Slice(0, 2)), (1 - active_r_cur) * f_r})),

                constr("stance_z",
                       {r_l_n, r_r_n, active_l, active_r},
                       cs::SX::vertcat({active_l * r_l_n(2), active_r * r_r_n(2)}))};
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