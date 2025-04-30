#include <atri/ocp/approx.hpp>

namespace atri {

raw_data::raw_data(expr_sets_ptr_t exprs) : exprs_(exprs) {
    for (size_t i = 0; i < field::num_sym; i++) {
        // auto &v = value_[i];
        // for (auto &e : exprs_->expr_[i]) {
        //     v.push_back(vector(e->dim_));
        // }
        value_[i].resize(exprs_->dim_[i]);
    }

    for (size_t i = __dyn; i < field::num_constr; i++) {
        if (exprs_->expr_[i].empty()) {
            continue;
        }
        size_t dim = exprs_->dim_[i];
        approx_[i].v_.resize(dim);
        for (size_t j = 0; j < field::num_prim; j++) {
            approx_[i].jac_[j].resize(dim, exprs_->dim_[j]);
        }
    }
    // cost hessian(store only half)
    for (size_t i = 0; i < field::num_prim; i++) {
        jac_[i].resize(exprs_->dim_[i]);
        for (size_t j = i; j < field::num_prim; j++) {
            hessian_[i][j].resize(exprs_->dim_[i], exprs_->dim_[j]);
        }
    }
}

sparse_approx_data::sparse_approx_data(raw_data &raw,
                                       std::vector<sym_ptr_t> in_args,
                                       approx &f)
    : v_(raw.approx_[f.field_].v_.segment(raw.exprs_->get_expr_start(f),
                                          f.dim_)) {
    for (size_t i = 0; i < in_args.size(); i++) {
        auto arg = in_args[i];
        in_args_.push_back(raw.get(arg));
    }
    size_t f_st = raw.exprs_->get_expr_start(f);
    if (f.order() >= approx_order::first) {
        for (size_t i = 0; i < in_args_.size(); i++) {
            jac_.push_back(raw.approx_[f.field_].jac_[in_args[i]->field_].block(
                f_st, raw.exprs_->get_expr_start(in_args[i]), f.dim_,
                in_args[i]->dim_));
        }
        assert(jac_.size() == in_args_.size());
    }
    if (f.order() >= approx_order::second) {
        size_t field_1, field_2;
        hess_.resize(in_args_.size());
        for (size_t i = 0; i < in_args_.size(); i++) {
            for (size_t j = i; j < in_args_.size(); j++) {
                /// @note order matches raw_data
                /// h[i][j] = h[j][i] if i, j in the same field or i < j
                /// otherwise only keep h[i][j], the other is empty
                field_1 = in_args[i]->field_;
                field_2 = in_args[j]->field_;
                if (field_1 <= field_2) {
                    hess_[i].push_back(raw.hessian_[field_1][field_2].block(
                        raw.exprs_->get_expr_start(in_args[i]),
                        raw.exprs_->get_expr_start(in_args[j]),
                        in_args[i]->dim_, in_args[j]->dim_));
                } else {
                    // this should be empty
                    hess_[i].push_back(raw.hessian_[field_1][field_2]);
                }
            }
        }
    }
}
sparse_approx_data_ptr_t approx::make_data(raw_data &raw) {
    auto data = sparse_approx_data_ptr_t();
    data.reset(new sparse_approx_data(raw, in_args_, *this));
    for (size_t i = 0; i < in_args_.size(); i++) {
        auto arg = in_args_[i];
    }
    setup_sparsity(data);
    return data;
}
} // namespace atri