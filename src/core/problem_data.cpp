#include <atri/ocp/core/approx_data.hpp>

namespace atri {

approx_data::approx_data(problem_ptr_t prob) : prob_(prob) {
    for (auto i : range_n(__dyn, field::num_constr)) {
        if (prob_->expr_[i].empty()) {
            continue;
        }
        size_t dim = prob_->dim_[i];
        approx_[i].v_.resize(dim);
        for (auto j : range(field::num_prim)) {
            approx_[i].jac_[j].resize(dim, prob_->dim_[j]);
        }
        // dual variables
        dual_[i].resize(prob_->dim_[i]);
    }
    // cost hessian(store only half)
    for (auto i : range(field::num_prim)) {
        for (auto j : range(i, field::num_prim)) {
            hessian_[i][j].resize(prob_->dim_[i], prob_->dim_[j]);
        }
        jac_[i].resize(prob_->dim_[i]);
    }
}
} // namespace atri