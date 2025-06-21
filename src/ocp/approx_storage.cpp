#include <atri/ocp/approx_storage.hpp>

namespace atri {

approx_storage::approx_storage(const problem_ptr_t &prob) : prob_(prob) {
    for (auto i : range_n(__dyn, field::num_constr)) {
        if (prob_->expr_[i].empty()) {
            continue;
        }
        size_t dim = prob_->dim_[i];
        approx_[i].v_.resize(dim);
        approx_[i].v_.setZero();
        for (auto j : range(field::num_prim)) {
            approx_[i].jac_[j].resize(dim, prob_->dim_[j]);
            approx_[i].jac_[j].setZero();
        }
        // dual variables
        dual_[i].resize(prob_->dim_[i]);
        dual_[i].setZero();
    }
    // cost val
    cost_ = 0;
    // cost hessian(store only half)
    for (auto i : range(field::num_prim)) {
        for (auto j : range(i, field::num_prim)) {
            hessian_[j][i].resize(prob_->dim_[j], prob_->dim_[i]);
            hessian_[j][i].setZero();
        }
        jac_[i].resize(prob_->dim_[i]);
        jac_[i].setZero();
    }
}
} // namespace atri