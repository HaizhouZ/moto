#include <atri/ocp/approx.hpp>

namespace atri {

primal_data::primal_data(expr_sets_ptr_t exprs) : exprs_(exprs) {
    for (size_t i = 0; i < field::num_sym; i++) {
        // auto &v = value_[i];
        // for (auto &e : exprs_->expr_[i]) {
        //     v.push_back(vector(e->dim_));
        // }
        value_[i].resize(exprs_->dim_[i]);
    }
}

approx_data::approx_data(primal_data &raw, std::vector<sym_ptr_t> in_args,
                         size_t dim, bool jac, bool hess) {
    for (size_t i = 0; i < in_args.size(); i++) {
        auto arg = in_args[i];
        in_args_.push_back(raw.get(arg));
    }
    v_.resize(dim);
    if (jac) {
        jac_.resize(in_args_.size());
        for (size_t i = 0; i < in_args_.size(); i++) {
            jac_[i].resize(dim, in_args[i]->dim_);
        }
    }
    if (hess) {
        if (dim == 1) {
            hess_.resize(in_args_.size());
            for (size_t i = 0; i < in_args_.size(); i++) {
                hess_[i].resize(in_args[i]->dim_, in_args[i]->dim_);
                // todo : cross hessian
            }
        }
    }
}

approx_data_ptr_t approx::make_data(primal_data &raw) {
    auto data = approx_data_ptr_t();
    data.reset(new approx_data(raw, in_args_, dim_,
                               order_ >= approx_order::first,
                               order_ >= approx_order::second));
    for (size_t i = 0; i < in_args_.size(); i++) {
        auto arg = in_args_[i];
    }
    setup_sparsity(data);
    return data;
}
} // namespace atri