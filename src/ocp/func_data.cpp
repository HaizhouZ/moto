#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
void sym::finalize_impl() {
    if (field_ == __x) {
        assert(dual_ && "dual pointer should not be null when field == __x");
    }
}
func_arg_map::func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f)
    : func_(f), impl_(shared), sym_uid_idx_(f.sym_uid_idx_) {
    auto &in_args = f.in_args();
    in_args_.reserve(in_args.size());
    for (auto &arg : in_args) {
        in_args_.push_back(primal[arg]);
    }
}
func_approx_data::func_approx_data(sym_data &primal,
                                   merit_data &raw,
                                   shared_data &shared,
                                   const generic_func &f)
    : func_arg_map(primal, shared, f),
      stored_(in_field(f.field(), merit_data::stored_constr_fields) || f.field() == __cost),
      v_data_(stored_ ? vector() : vector::Zero(f.dim())),
      v_(stored_ ? (f.field() == __cost ? vector_ref(mapped_vector(&raw.cost_, 1))
                                        : raw.approx_[f.field()].v_.segment(raw.prob_->get_expr_start(f), f.dim()))
                 : vector_ref(v_data_)),
      merit_data_(&raw) {
    auto &in_args = f.in_args();
    size_t f_st = raw.prob_->get_expr_start(f);
    // for non-cost
    if (f.order() >= approx_order::first) {
        // bind merit jacobian
        merit_jac_.reserve(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim) {
                merit_jac_.push_back(raw.jac_[in_args[i]->field()].segment(
                    raw.prob_->get_expr_start(in_args[i]), in_args[i]->dim()));
            } else { // useless
                static row_vector empty;
                merit_jac_.push_back(empty);
            }
        }
        // bind approx jacobian
        if (f.field() != __cost) {
            if (!stored_)
                jac_data_.reserve(in_args.size());
            jac_.reserve(in_args.size());
            for (size_t i : range(in_args.size())) {
                if (in_args[i]->field() < field::num_prim) {
                    if (!stored_) {
                        jac_data_.emplace_back(f.dim(), in_args[i]->dim());
                        jac_data_.back().setZero();
                    }
                    if (stored_) {
                        jac_.emplace_back(raw.approx_[f.field()].jac_[in_args[i]->field()].block(
                            f_st, raw.prob_->get_expr_start(in_args[i]),
                            f.dim(), in_args[i]->dim()));
                    } else {
                        jac_.emplace_back(jac_data_.back());
                    }
                } else { // useless
                    static matrix empty;
                    if (!stored_)
                        jac_data_.emplace_back();
                    jac_.push_back(empty);
                }
            }
        } else {
            jac_.reserve(merit_jac_.size());
            jac_.assign(merit_jac_.begin(), merit_jac_.end());
        }
    }
    setup_hessian(raw);
}
void func_approx_data::setup_hessian(merit_data &raw) {
    auto &f = func_;
    auto &in_args = f.in_args();
    if (f.order() >= approx_order::second || in_field(f.field(), ineq_soft_constr_fields)) {
        size_t field_1, field_2;
        merit_hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim) {
                merit_hess_[i].reserve(in_args_.size());
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field();
                    field_2 = in_args[j]->field();
                    if (field_2 < field::num_prim) {
                        /// @note order matches merit_data
                        /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                        /// otherwise only keep h[i][j] (empty)
                        if (field_1 >= field_2) {
                            merit_hess_[i].push_back(raw.hessian_[field_1][field_2].block(
                                raw.prob_->get_expr_start(in_args[i]),
                                raw.prob_->get_expr_start(in_args[j]),
                                in_args[i]->dim(), in_args[j]->dim()));
                            continue;
                        }
                    }
                    // this should be empty. do this anyway to make the shape of merit_hess_ right
                    static matrix empty;
                    merit_hess_[i].push_back(empty);
                }
            }
        }
        merit_hess_.shrink_to_fit();
    }
}
} // namespace moto