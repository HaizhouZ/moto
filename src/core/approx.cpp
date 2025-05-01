#include <atri/ocp/core/approx.hpp>
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

sparse_approx_data::sparse_approx_data(problem_data *raw,
                                       std::vector<sym_ptr_t> in_args,
                                       approx *f)
    : v_(raw->approx_[f->field_].v_.segment(raw->prob_->get_expr_start(*f),
                                            f->dim_)) {
    for (size_t i = 0; i < in_args.size(); i++) {
        auto arg = in_args[i];
        in_args_.push_back(raw->get(arg));
    }
    size_t f_st = raw->prob_->get_expr_start(*f);
    // for non-cost
    if (f->field_ - __dyn < field::num_constr) {
        if (f->order() >= approx_order::first) {
            for (size_t i = 0; i < in_args_.size(); i++) {
                jac_.push_back(raw->approx_[f->field_].jac_[in_args[i]->field_].block(
                    f_st, raw->prob_->get_expr_start(in_args[i]),
                    f->dim_, in_args[i]->dim_));
            }
            assert(jac_.size() == in_args_.size());
        }
    } else { // for cost
        for (size_t i = 0; i < in_args_.size(); i++) {
            jac_.push_back(raw->jac_[in_args[i]->field_].segment(
                raw->prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
        }
    }

    if (f->order() >= approx_order::second) {
        size_t field_1, field_2;
        hess_.resize(in_args_.size());
        for (size_t i = 0; i < in_args_.size(); i++) {
            for (size_t j = 0; j < in_args_.size(); j++) {
                /// @note order matches problem_data
                /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                /// otherwise only keep h[i][j] (empty)
                field_1 = in_args[i]->field_;
                field_2 = in_args[j]->field_;
                if (field_1 <= field_2) {
                    hess_[i].push_back(raw->hessian_[field_1][field_2].block(
                        raw->prob_->get_expr_start(in_args[i]),
                        raw->prob_->get_expr_start(in_args[j]),
                        in_args[i]->dim_, in_args[j]->dim_));
                } else {
                    // this should be empty
                    hess_[i].push_back(raw->hessian_[field_1][field_2]);
                }
            }
        }
    }
}
sparse_approx_data_ptr_t approx::make_data(problem_data *raw) {
    auto data = sparse_approx_data_ptr_t();
    data.reset(new sparse_approx_data(raw, in_args_, this));
    for (size_t i = 0; i < in_args_.size(); i++) {
        auto arg = in_args_[i];
    }
    setup_sparsity(data);
    return data;
}
} // namespace atri