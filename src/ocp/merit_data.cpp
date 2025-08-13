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
            approx_[i].state_jac_.resize(prob_->exprs(i).size());
            approx_[i].state_jac_.shrink_to_fit();
            // for (auto &jac : approx_[i].state_jac_) {
            //     for (auto f : state_fields) {
            //         jac[f].resize(n_dyn);
            //         jac[f].shrink_to_fit();
            //     }
            // }
            approx_[i].non_state_jac_.resize(prob_->exprs(i).size());
            approx_[i].non_state_jac_.shrink_to_fit();
        }
        // dual variables
        dual_[i].resize(prob_->dim(i));
        dual_[i].setZero();
    }
    // dynamics data
    dynamics_data_.reserve(n_dyn);
    for (expr &ex : prob_->exprs(__dyn)) {
        dynamics_data d;
        d.v_.resize(ex.dim());
        d.v_.setZero();
        // leave this free for now, will be filled in later
        dynamics_data_.emplace_back(std::move(d));
    }
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