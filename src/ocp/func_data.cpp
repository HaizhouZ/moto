#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/func_data.hpp>

namespace moto {
func_arg_map::func_arg_map(sym_data &primal, shared_data &shared, impl::func &f)
    : func_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_) {
    auto &in_args = f.in_args();
    in_args_.reserve(in_args.size());
    for (auto &arg : in_args) {
        in_args_.push_back(primal[arg]);
    }
}
func_arg_map::func_arg_map(std::vector<vector_ref> &&primal, shared_data &shared, impl::func &f)
    : in_args_(std::move(primal)), func_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_) {
}
func_approx_map::func_approx_map(sym_data &primal,
                                 dense_approx_data &raw,
                                 shared_data &shared,
                                 impl::func &f)
    : func_arg_map(primal, shared, f),
      v_(f.field_ == __cost
             ? vector_ref(mapped_vector(&raw.cost_, 1))
             : raw.approx_[f.field_].v_.segment(raw.prob_->get_expr_start(f), f.dim_)) {
    auto &in_args = f.in_args();
    size_t f_st = raw.prob_->get_expr_start(f);
    // for non-cost
    if (f.field_ - __dyn < field::num_constr) {
        if (f.order() >= approx_order::first) {
            jac_.reserve(in_args_.size());
            for (size_t i : range(in_args_.size())) {
                if (in_args[i]->field_ < field::num_prim) {
                    jac_.push_back(raw.approx_[f.field_].jac_[in_args[i]->field_].block(
                        f_st, raw.prob_->get_expr_start(in_args[i]),
                        f.dim_, in_args[i]->dim_));
                } else { // useless
                    static matrix empty;
                    jac_.push_back(empty);
                }
            }
        }
    } else { // for cost
        jac_.reserve(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field_ < field::num_prim) {
                jac_.push_back(raw.jac_[in_args[i]->field_].segment(
                    raw.prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
            } else { // useless
                static matrix empty;
                jac_.push_back(empty);
            }
        }
    }
    setup_hessian(raw);
}
func_approx_map::func_approx_map(sym_data &primal,
                                 vector_ref v,
                                 std::vector<matrix_ref> &&jac,
                                 shared_data &shared,
                                 impl::func &f)
    : v_(v), jac_(jac), func_arg_map(primal, shared, f) {
}
void func_approx_map::setup_hessian(dense_approx_data &raw) {
    auto &f = func_;
    auto &in_args = f.in_args();
    if (f.order() >= approx_order::second || f.field_ - __dyn < field::num_constr) {
        size_t field_1, field_2;
        hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field_ < field::num_prim) {
                hess_[i].reserve(in_args_.size());
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field_;
                    field_2 = in_args[j]->field_;
                    if (field_2 < field::num_prim) {
                        /// @note order matches dense_approx_data
                        /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                        /// otherwise only keep h[i][j] (empty)
                        if (field_1 >= field_2) {
                            hess_[i].push_back(raw.hessian_[field_1][field_2].block(
                                raw.prob_->get_expr_start(in_args[i]),
                                raw.prob_->get_expr_start(in_args[j]),
                                in_args[i]->dim_, in_args[j]->dim_));
                            continue;
                        }
                    }
                    // this should be empty. do this anyway to make the shape of hess_ right
                    static matrix empty;
                    hess_[i].push_back(empty);
                }
            }
        }
    }
}
} // namespace moto