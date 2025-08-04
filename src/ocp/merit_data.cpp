#include <moto/ocp/impl/merit_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {

merit_data::merit_data(ocp *prob) : prob_(prob) {
    prob->wait_until_ready();
    for (auto i : range_n(__dyn, field::num_constr)) {
        if (prob_->exprs(i).empty()) {
            continue;
        }
        size_t dim = prob_->dim(i);
        if (in_field(i, merit_data::stored_constr_fields)) {
            approx_[i].v_.resize(dim);
            approx_[i].v_.setZero();
            for (auto j : range(field::num_prim)) {
                approx_[i].jac_[j].resize(dim, prob_->dim(j));
                approx_[i].jac_[j].setZero();
            }
        }
        // dual variables
        dual_[i].resize(prob_->dim(i));
        dual_[i].setZero();
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
}
} // namespace moto