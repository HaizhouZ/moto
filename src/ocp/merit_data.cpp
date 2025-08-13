#include <moto/ocp/impl/merit_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

merit_data::merit_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    size_t n_dyn = prob_->exprs(__dyn).size();
    for (auto i : constr_fields) {
        if (prob_->exprs(i).empty()) {
            continue;
        }
        size_t dim = prob_->dim(i);
        if (in_field(i, merit_data::stored_constr_fields)) {
            approx_[i].v_.resize(dim);
            approx_[i].v_.setZero();
            for (auto f : primal_fields) {
                approx_[i].jac_[f].resize(dim, prob_->dim(f));
            }
        }
        // dual variables
        dual_[i].resize(prob_->dim(i));
        dual_[i].setZero();
    }
    // dynamics data
    dynamics_data_.proj_f_res_.resize(prob_->dim(__dyn));
    dynamics_data_.proj_f_res_.setZero();
    dynamics_data_.proj_f_x_.resize(prob_->dim(__dyn), prob_->dim(__x));
    dynamics_data_.proj_f_u_.resize(prob_->dim(__dyn), prob_->dim(__u));
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
            hessian_[j][i].resize(prob_->dim(j), prob_->dim(i));
            hessian_[j][i].setZero();
        }
        jac_[i].resize(prob_->dim(i));
        jac_[i].setZero();
        jac_modification_[i].resize(prob_->dim(i));
        jac_modification_[i].setZero();
    }
    hessian_modification_ = hessian_; // same size
    for (auto f : primal_fields) {
        res_stat_[f].resize(prob_->dim(f));
        res_stat_[f].setZero();
    }
}
} // namespace moto