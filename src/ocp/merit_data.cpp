#include <moto/ocp/impl/lag_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

lag_data::lag_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
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
            lag_hess_[j][i].resize(prob_->tdim(j), prob_->tdim(i));
            // lag_hess_[j][i].setZero();
        }
        lag_hess_[i][i].insert<sparsity::diag>(0, 0, prob_->tdim(i));
        cost_jac_[i].resize(prob_->tdim(i));
        cost_jac_[i].setZero();
        lag_jac_[i].resize(prob_->tdim(i));
        lag_jac_[i].setZero();
        lag_jac_corr_[i].resize(prob_->tdim(i));
        lag_jac_corr_[i].setZero();
    }
    hessian_modification_ = lag_hess_; // same size
}
} // namespace moto
