#include <moto/core/external_function.hpp>
#include <moto/ocp/approx_storage.hpp>
#include <moto/ocp/func.hpp>
#include <moto/ocp/sym_data.hpp>

namespace moto {
sp_arg_map::sp_arg_map(sym_data &primal, shared_data &shared, func_impl &f)
    : f_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_) {
    auto &in_args = f.in_args();
    for (auto &arg : in_args) {
        in_args_.push_back(primal[arg]);
    }
}
sp_arg_map::sp_arg_map(std::vector<vector_ref> &&primal, shared_data &shared, func_impl &f)
    : in_args_(std::move(primal)), f_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_) {
}
sp_approx_map::sp_approx_map(sym_data &primal,
                             approx_storage &raw,
                             shared_data &shared,
                             func_impl &f)
    : sp_arg_map(primal, shared, f),
      v_(f.field_ == __cost
             ? vector_ref(mapped_vector(&raw.cost_, 1))
             : raw.approx_[f.field_].v_.segment(raw.prob_->get_expr_start(f), f.dim_)) {
    auto &in_args = f.in_args();
    size_t f_st = raw.prob_->get_expr_start(f);
    // for non-cost
    if (f.field_ - __dyn < field::num_constr) {
        if (f.order() >= approx_order::first) {
            for (size_t i : range(in_args_.size())) {
                if (in_args[i]->field_ < field::num_prim) {
                    jac_.push_back(raw.approx_[f.field_].jac_[in_args[i]->field_].block(
                        f_st, raw.prob_->get_expr_start(in_args[i]),
                        f.dim_, in_args[i]->dim_));
                } else // useless
                    jac_.push_back(raw.approx_[f.field_].jac_[in_args[i]->field_]);
            }
        }
    } else { // for cost
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

    if (f.order() >= approx_order::second) {
        size_t field_1, field_2;
        hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field_ < field::num_prim) {
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field_;
                    field_2 = in_args[j]->field_;
                    if (field_2 < field::num_prim) {
                        /// @note order matches approx_storage
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
sp_approx_map::sp_approx_map(sym_data &primal,
                             vector_ref v,
                             const std::vector<matrix_ref> &jac,
                             shared_data &shared,
                             func_impl &f)
    : v_(v), jac_(jac), sp_arg_map(primal, shared, f) {
}
shared_data::shared_data(const ocp_ptr_t &prob, sym_data &primal) {
    for (const auto &expr : prob->expr_[__pre_comp]) {
        data_.try_emplace(expr->uid_, std::static_pointer_cast<func_impl>(expr)->make_data(primal, *this));
    }
    for (const auto &expr : prob->expr_[__usr_func]) {
        data_.try_emplace(expr->uid_, std::static_pointer_cast<func_impl>(expr)->make_data(primal, *this));
    }
}

sp_approx_map_ptr_t func_impl::make_approx_data_mapping(sym_data &primal, approx_storage &raw, shared_data &shared) {
    if (field_ - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("make_approx_data_mapping cannot be called for func {} type {}",
                                             name_, magic_enum::enum_name(field_)));
    if (in_args_.empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name_, magic_enum::enum_name(field_)));
    ;
    auto approx_data = std::make_unique<sp_approx_map>(primal, raw, shared, *this);
    setup_sparsity(*approx_data);
    return approx_data;
}
void func_impl::load_external(const std::string &path) {
    auto funcs = load_approx(name_, true, order() >= approx_order::first, order() >= approx_order::second);
    value = [eval = funcs[0]](sp_approx_map &d) {
        eval.invoke(d.in_args(), d.v_);
    };
    jacobian = [jac = funcs[1]](sp_approx_map &d) {
        jac.invoke(d.in_args(), d.jac_);
    };

    hessian = [hess = funcs[2]](sp_approx_map &d) {
        hess.invoke(d.in_args(), d.hess_);
    };
}
} // namespace moto