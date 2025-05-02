#include <atri/ocp/core/problem_data.hpp>

namespace atri {

problem_data::problem_data(problem_ptr_t prob) : prob_(prob) {
    for (size_t i = 0; i < field::num_sym; i++) {
        value_[i].resize(prob_->dim_[i]);
    }

    for (size_t i = __dyn, cnt = 0; cnt < field::num_constr; i++, cnt++) {
        if (prob_->expr_[i].empty()) {
            continue;
        }
        size_t dim = prob_->dim_[i];
        approx_[i].v_.resize(dim);
        for (size_t j = 0; j < field::num_prim; j++) {
            approx_[i].jac_[j].resize(dim, prob_->dim_[j]);
        }
    }
    // cost hessian(store only half)
    for (size_t i = 0; i < field::num_prim; i++) {
        for (size_t j = i; j < field::num_prim; j++) {
            hessian_[i][j].resize(prob_->dim_[i], prob_->dim_[j]);
        }
        jac_[i].resize(prob_->dim_[i]);
    }
}
}