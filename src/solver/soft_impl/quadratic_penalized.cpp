// #include <moto/solver/soft_constr/quadratic_penalized.hpp>

// namespace moto {
// namespace solver {

// quadratic_penalized_constr::approx_data::approx_data(base::approx_data &&rhs)
//     : base::approx_data(std::move(rhs)) {
//     slack_.resize(func_.dim());
//     slack_.setConstant(1e-6);
//     diag_scaling.resize(func_.dim());
//     diag_scaling.setZero();
//     scaled_res_.resize(func_.dim());
//     scaled_res_.setZero();
//     reg_.resize(func_.dim());
//     reg_.setZero();
//     reg_T_inv_.resize(func_.dim());
//     reg_T_inv_.setZero();
//     active_.resize(func_.dim());
//     active_.setZero();
// }
// void quadratic_penalized_constr::initialize(data_map_t &data) const {
//     // dont call value_impl here
//     auto &d = data.as<approx_data>();
//     // dont do d.g_ = d.v_; cuz already set in value_impl
//     d.slack_ = (-d.g_).cwiseMax(1e-2); // clip
//     d.v_ = d.g_ + d.slack_;            // r_g = g_ + slack
// }
// void quadratic_penalized_constr::finalize_newton_step(data_map_t &data) const {
//     auto &d = data.as<approx_data>();
//     size_t arg_idx = 0;
    

// }
// void quadratic_penalized_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
//     auto &d = data.as<approx_data>();
//     auto &ls_cfg = cfg->as<solver::linesearch_config>();
//     assert(!d.quad_penalized_constr_cfg->quad_penalized_constr_computing_affine_step() && "quad_penalized_constr affine step computation not ended");
//     d.d_slack_.array() *= ls_cfg.alpha_primal;
//     d.d_multiplier_.array() *= ls_cfg.alpha_dual;
//     d.slack_.array() += d.d_slack_.array();
//     d.multiplier_.array() += d.d_multiplier_.array();
// }
// void quadratic_penalized_constr::value_impl(func_approx_data &data) const {
//     base::value_impl(data);
//     auto &d = data.as<approx_data>();
//     d.g_ = d.v_;            //.cwiseMin(-d.reg_);
//     d.v_ = d.g_ + d.slack_; // r_g = g_ + slack
//     d.reg_.setConstant(1e-8);
// }
// void quadratic_penalized_constr::jacobian_impl(func_approx_data &data) const {
//     base::jacobian_impl(data);
//     auto &d = data.as<approx_data>();

//     // setup T^{-1} N
//     d.reg_T_inv_.array() = d.slack_.array() + d.reg_.array() * d.multiplier_.array();
//     d.diag_scaling.array() = d.active_.array() * d.multiplier_.array() / d.reg_T_inv_.array();
//     d.scaled_res_.array() = d.diag_scaling.array() * d.g_.array();
//     if (!d.quad_penalized_constr_cfg->quad_penalized_constr_enable_affine_step()) {
//         // if we are not in the affine step mode, we need to update the scaled residual with mu
//         d.scaled_res_.array() += d.quad_penalized_constr_cfg->mu / d.reg_T_inv_.array();
//     }
//     d.scaled_res_.array() *= d.active_.array(); // only active constraints contribute to the scaled residual
//     propagate_jacobian(d);
//     propagate_hessian(d);
// }
// void quadratic_penalized_constr::propagate_jacobian(func_approx_data &data) const {
//     size_t j_idx = 0;
//     bool nan_found = false;
//     auto &d = data.as<approx_data>();
//     for (auto &j : d.jac_) {
//         if (j.size() != 0) {
//             d.jac_modification_[j_idx].noalias() += d.scaled_res_.transpose() * j;
//             if (d.jac_modification_[j_idx].hasNaN()) {
//                 nan_found = true;
//                 fmt::print("--------------------\n");
//                 fmt::print("constraint name: {}\n", d.func_.name());
//                 for (sym &arg : d.func_.in_args()) {
//                     fmt::print("arg: {}: {}\n", arg.name(), d[arg].transpose());
//                 }
//                 fmt::print("jac: \n{:.3}\n", j);
//                 fmt::print("g: {:.3}\n", d.g_.transpose());
//                 fmt::print("slack: {:.3}\n", d.slack_.transpose());
//                 fmt::print("multiplier: {:.3}\n", d.multiplier_.transpose());
//                 fmt::print("diag_scaling: {:.3}\n", d.diag_scaling.transpose());
//                 fmt::print("scaled_res: {:.3}\n", d.scaled_res_.transpose());
//                 fmt::print("jac modification: {:.3}\n", d.jac_modification_[j_idx]);
//                 fmt::print("NaN in jac modification[{}]\n", j_idx);
//             }
//         }
//         j_idx++;
//     }
//     if (nan_found) {
//         fmt::print("NaN found in jacobian modification for constraint: {}\n", d.func_.name());
//         throw std::runtime_error("NaN found in jacobian modification");
//     }
// }
// void quadratic_penalized_constr::propagate_hessian(func_approx_data &d) const {
//     // modification of hessian
//     size_t outer_idx = 0;
//     for (auto &outer : d.merit_hess_) {
//         size_t inner_idx = 0;
//         if (outer.size()) { // skip empty hess
//             for (auto &inner : outer) {
//                 if (inner.size() != 0) {
//                     inner.noalias() += d.jac_[outer_idx].transpose() * d.as<approx_data>().diag_scaling.asDiagonal() * d.jac_[inner_idx];
//                     assert((d.as<approx_data>().diag_scaling.array() > 0).all() && "diag_scaling must be positive");
//                     if (inner.hasNaN()) {
//                         fmt::print("NaN in hess[{}][{}]\n", outer_idx, inner_idx);
//                     }
//                 }
//                 inner_idx++;
//             }
//         }
//         outer_idx++;
//     }
// }

// } // namespace solver
// } // namespace moto