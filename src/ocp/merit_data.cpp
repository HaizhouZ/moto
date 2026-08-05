#include <moto/ocp/impl/lag_data.hpp>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

lag_data::lag_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    const auto &profile = prob_->linear_profile();
    size_t n_dyn = prob_->exprs(__dyn).size();
    for (auto i : constr_fields) {
        if (prob_->exprs(i).empty()) {
            continue;
        }
        size_t dim = prob_->dim(i);
        if (in_field(i, lag_data::stored_constr_fields)) {
            approx_[i].v_.resize(dim);
            approx_[i].v_.setZero();
            for (auto f : primal_fields) {
                approx_[i].jac_[f].resize(dim, prob_->tdim(f));
                approx_[i].jac_[f].plan(
                    profile.get(linear_target::jacobian, i, f));
                // fmt::println("prob {} lag_data: approx jacobian for constr field {} w.r.t. primal field {} has dim {}x{}",
                //              prob_->uid(), field::name(i), field::name(f), dim, prob_->tdim(f));
            }
        }
        // dual variables
        dual_[i].resize(prob_->dim(i));
        dual_[i].setZero();
    }
    // dynamics data
    dynamics_data_.proj_f_res_.resize(prob_->dim(__dyn));
    dynamics_data_.proj_f_res_.setZero();
    dynamics_data_.proj_f_x_.resize(prob_->dim(__dyn), prob_->tdim(__x));
    dynamics_data_.proj_f_u_.resize(prob_->dim(__dyn), prob_->tdim(__u));
    std::vector<sparse_block_spec> projected_x, projected_u;
    for (const auto &entry : prob_->exprs(__dyn)) {
        const auto *dyn = dynamic_cast<const generic_dynamics *>(entry.get());
        if (!dyn) continue;
        const size_t f_st = prob_->get_expr_start(*dyn);
        for (const auto &[argument, block] : dyn->projected_panel_sparsity()) {
            const sym &arg = dyn->in_args(argument);
            if (!prob_->is_active(arg)) continue;
            auto &target = arg.field() == __x ? projected_x : projected_u;
            target.push_back({f_st + block.row_offset,
                              prob_->get_expr_start_tangent(arg) +
                                  block.col_offset,
                              block.rows, block.cols, block.pattern});
        }
    }
    const auto plan_projected = [](sparse_matrix &target,
                                   const std::vector<sparse_block_spec> &blocks) {
        if (blocks.empty()) return;
        auto layout = make_sparse_layout_plan(blocks);
        layout.pack_diagonal_storage = true;
        target.plan(layout);
    };
    plan_projected(dynamics_data_.proj_f_x_, projected_x);
    plan_projected(dynamics_data_.proj_f_u_, projected_u);
    // complementarity
    for (auto f : ineq_constr_fields) {
        comp_[f].resize(prob_->dim(f));
        comp_[f].setZero();
    }
    // cost val
    cost_ = 0;
    // cost hessian(store only half)
    for (auto i : range(field::num_prim)) {
        for (auto j : range(i, field::num_prim)) {
            const auto fi = static_cast<field_t>(i);
            const auto fj = static_cast<field_t>(j);
            lag_hess_[j][i].resize(prob_->tdim(j), prob_->tdim(i));
            lag_hess_[j][i].plan(
                profile.get(linear_target::lag_hessian, fj, fi));
            hessian_modification_[j][i].resize(prob_->tdim(j), prob_->tdim(i));
            hessian_modification_[j][i].plan(
                profile.get(linear_target::hessian_modification, fj, fi));
            // lag_hess_[j][i].setZero();
        }
        lag_hess_[i][i].bind(0, 0, prob_->tdim(i), prob_->tdim(i),
                             sparsity::diag);
        hessian_modification_[i][i].bind(
            0, 0, prob_->tdim(i), prob_->tdim(i), sparsity::diag);
        cost_jac_[i].resize(prob_->tdim(i));
        cost_jac_[i].setZero();
        lag_jac_[i].resize(prob_->tdim(i));
        lag_jac_[i].setZero();
        lag_jac_corr_[i].resize(prob_->tdim(i));
        lag_jac_corr_[i].setZero();
    }
}
} // namespace moto
