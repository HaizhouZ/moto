#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
func_arg_map::func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f)
    : func_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_), primal_(&primal) {
    auto &in_args = f.in_args();
    in_args_.reserve(in_args.size());
    for (auto &arg : in_args) {
        static vector empty;
        if (problem()->is_active(arg))
            in_args_.push_back(primal[arg]);
        else
            in_args_.push_back(empty);
    }
}

vector_ref get_value_ref(const generic_func &f, lag_data &raw) {
    if (f.field() == __cost) {
        return vector_ref(mapped_vector(&raw.cost_, 1));
    } else if (in_field(f.field(), lag_data::stored_constr_fields)) {
        return raw.approx_[f.field()].v_.segment(raw.prob_->get_expr_start(f), f.dim());
    } else {
        throw std::runtime_error(fmt::format("Function {} in field {} with uid {} does not have stored value",
                                             f.name(), f.field(), f.uid()));
    }
}

func_approx_data::func_approx_data(sym_data &primal,
                                   lag_data &raw,
                                   shared_data &shared,
                                   const generic_func &f)
    : func_arg_map(primal, shared, f), v_(get_value_ref(f, raw)), lag_data_(&raw) {
    auto &in_args = f.in_args();
    if (f.order() >= approx_order::first) {
        jac_.reserve(in_args_.size());
        const auto f_field = func_.field();
        auto *prob = raw.prob_;
        for (size_t i : range(in_args_.size())) {
            auto arg = in_args[i];
            if (arg->field() < field::num_prim && prob->is_active(arg) && f_field != __dyn) {
                if (f_field == __cost) {
                    jac_.push_back(prob->extract_row_tangent(raw.cost_jac_[arg->field()], arg));
                    continue;
                } else if (in_field(f_field, lag_data::stored_constr_fields)) {
                    const auto sp = func_.jac_sparsity()[i];
                    const auto f_st = prob->get_expr_start(func_);
                    auto &jac = lag_data_->approx_[f_field].jac_[arg->field()];
                    const auto r_st = f_st + sp.row_offset;
                    const auto c_st = prob->get_expr_start_tangent(arg) + sp.col_offset;
                    jac_.push_back(matrix_ref(jac.insert(r_st, c_st, sp.rows, sp.cols, sp.pattern)));
                    continue;
                }
            }
            static matrix empty;
            jac_.push_back(empty);
        }
    }
    setup_hessian();
}

void func_approx_data::setup_hessian() {
    auto &f = func_;
    auto &in_args = f.in_args();
    assert(lag_data_ != nullptr && "lag_data_ should not be null");
    auto &raw = *lag_data_;
    bool is_ineq_soft = in_field(f.field(), ineq_soft_constr_fields);
    if (f.order() >= approx_order::second || is_ineq_soft) {
        size_t field_1, field_2;
        auto *hessian = f.field() == __cost ? &raw.lag_hess_ : &raw.hessian_modification_;
        lag_hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim) {
                lag_hess_[i].reserve(in_args_.size());
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field();
                    field_2 = in_args[j]->field();
                    if (raw.prob_->is_active(in_args[i]) &&
                        field_2 < field::num_prim &&
                        raw.prob_->is_active(in_args[j])) {
                        /// @note order matches lag_data
                        /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                        /// otherwise only keep h[i][j] (empty)
                        if (func_.hess_sp_[i][j].pattern == sparsity::unknown) {
                            goto BIND_EMPTY_HESS;
                        } else if (field_1 >= field_2) {
                            const auto &hess_sp = func_.hess_sp_[i][j];
                            lag_hess_[i].push_back((*hessian)[field_1][field_2].insert(
                                raw.prob_->get_expr_start_tangent(in_args[i]) + hess_sp.row_offset,
                                raw.prob_->get_expr_start_tangent(in_args[j]) + hess_sp.col_offset,
                                hess_sp.rows, hess_sp.cols, hess_sp.pattern));
                            continue;
                        }
                    }
                BIND_EMPTY_HESS:
                    // this should be empty. do this anyway to make the shape of lag_hess_ right
                    static matrix empty;
                    lag_hess_[i].push_back(empty);
                }
            }
        }
        lag_hess_.shrink_to_fit();
    }
}

bool func_approx_data::has_jacobian_block(size_t arg_idx) const {
    return arg_idx < jac_.size() && jac_[arg_idx].size() != 0;
}
} // namespace moto
