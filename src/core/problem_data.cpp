#include <atri/ocp/core/approx_data.hpp>

namespace atri {

approx_data::approx_data(problem_ptr_t prob) : prob_(prob) {
    for (size_t i = __dyn, cnt = 0; cnt < field::num_constr; i++, cnt++) {
        if (prob_->expr_[i].empty()) {
            continue;
        }
        size_t dim = prob_->dim_[i];
        approx_[i].v_.resize(dim);
        for (size_t j = 0; j < field::num_prim; j++) {
            approx_[i].jac_[j].resize(dim, prob_->dim_[j]);
        }
        // dual variables
        dual_[i].resize(prob_->dim_[i]);
    }
    // cost hessian(store only half)
    for (size_t i = 0; i < field::num_prim; i++) {
        for (size_t j = i; j < field::num_prim; j++) {
            hessian_[i][j].resize(prob_->dim_[i], prob_->dim_[j]);
        }
        jac_[i].resize(prob_->dim_[i]);
    }
}
} // namespace atri