#include <atri/ocp/core/approx.hpp>
#include <atri/ocp/core/approx_data.hpp>
#include <atri/ocp/core/sym_data.hpp>

namespace atri {

sparse_approx_data::sparse_approx_data(sym_data *primal,
                                       approx_data *raw,
                                       std::vector<sym_ptr_t> in_args,
                                       approx *f)
    : v_(raw->approx_[f->field_].v_.segment(raw->prob_->get_expr_start(*f),
                                            f->dim_)),
      sym_uid_idx_(f->sym_uid_idx_) {
    for (size_t i = 0; i < in_args.size(); i++) {
        auto arg = in_args[i];
        in_args_.push_back(primal->get(arg));
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
                /// @note order matches approx_data
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
sparse_approx_data_ptr_t approx::make_data(sym_data *primal, approx_data *raw) {
    assert(!in_args_.empty());
    auto data = sparse_approx_data_ptr_t();
    data.reset(new sparse_approx_data(primal, raw, in_args_, this));
    for (size_t i = 0; i < in_args_.size(); i++) {
        auto arg = in_args_[i];
    }
    setup_sparsity(*data);
    return data;
}
} // namespace atri