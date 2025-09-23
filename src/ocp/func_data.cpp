#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
sym::sym(const std::string &name, size_t dim, field_t type, default_val_t default_val)
    : expr(name, dim, type), cs::SX(cs::SX::sym(name, dim)) {
    if (!(size_t(type) <= field::num_sym || type == __usr_var))
        throw std::runtime_error(fmt::format("Invalid field {} for symbolic variable {}", type, name));
    integrator_ = [](vector_ref x, vector_ref dx, vector_ref out, scalar_t alpha) {
        out = x + alpha * dx;
    };
    set_default_value(default_val);
}
void sym::set_default_value(const default_val_t &default_val) {
    if (std::holds_alternative<vector>(default_val)) {
        auto &v = std::get<vector>(default_val);
        if (v.size() != dim())
            throw std::runtime_error(fmt::format("default value size mismatch for sym {} in field {}, expected {}, got {}",
                                                 name(), field::name(field()), dim(), v.size()));
        default_value_ = std::move(v);
    } else if (std::holds_alternative<scalar_t>(default_val)) {
        default_value_ = vector::Constant(dim(), std::get<scalar_t>(default_val));
    } /// leave empty
}
void sym::finalize_impl() {
    if (field_ == __x && !bool(dual_)) {
        throw std::runtime_error("dual pointer should not be null when field == __x");
    } else if (field_ == __y && !bool(dual_)) {
        throw std::runtime_error("dual pointer should be null when field == __y");
    }
    set_ready_status(true);
}
func_arg_map::func_arg_map(sym_data &primal, shared_data &shared, const generic_func &f)
    : func_(f), shared_(shared), sym_uid_idx_(f.sym_uid_idx_), primal_(&primal) {
    auto &in_args = f.in_args();
    in_args_.reserve(in_args.size());
    for (auto &arg : in_args) {
        static vector empty;
        if (problem()->contains(arg))
            in_args_.push_back(primal[arg]);
        else
            in_args_.push_back(empty);
    }
}

vector_ref get_value_ref(const generic_func &f, merit_data &raw) {
    if (f.field() == __cost) {
        return vector_ref(mapped_vector(&raw.cost_, 1));
    } else if (in_field(f.field(), merit_data::stored_constr_fields)) {
        return raw.approx_[f.field()].v_.segment(raw.prob_->get_expr_start(f), f.dim());
    } else {
        throw std::runtime_error(fmt::format("Function {} in field {} with uid {} does not have stored value",
                                             f.name(), f.field(), f.uid()));
    }
}

func_approx_data::func_approx_data(sym_data &primal,
                                   merit_data &raw,
                                   shared_data &shared,
                                   const generic_func &f)
    : func_arg_map(primal, shared, f), v_(get_value_ref(f, raw)), merit_data_(&raw) {
    auto &in_args = f.in_args();
    size_t f_st = raw.prob_->get_expr_start(f);
    // for non-cost
    if (f.order() >= approx_order::first) {
        // bind merit jacobian
        merit_jac_.reserve(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim && raw.prob_->contains(in_args[i])) {
                merit_jac_.push_back(raw.prob_->extract_tangent(raw.jac_[in_args[i]->field()], in_args[i]));
            } else { // useless
                static row_vector empty;
                merit_jac_.push_back(empty);
            }
        }
        // bind approx jacobian
        if (f.field() != __cost) {
            static matrix empty;
            jac_.assign(in_args_.size(), empty);
        } else {
            jac_.reserve(merit_jac_.size());
            jac_.assign(merit_jac_.begin(), merit_jac_.end());
        }
    }
    setup_hessian();
}

void func_approx_data::setup_hessian() {
    auto &f = func_;
    auto &in_args = f.in_args();
    assert(merit_data_ != nullptr && "merit_data_ should not be null");
    auto &raw = *merit_data_;
    if (f.order() >= approx_order::second || in_field(f.field(), ineq_soft_constr_fields)) {
        size_t field_1, field_2;
        auto *hessian = f.field() == __cost ? &raw.hessian_ : &raw.hessian_modification_;
        merit_hess_.resize(in_args_.size());
        for (size_t i : range(in_args_.size())) {
            if (in_args[i]->field() < field::num_prim) {
                merit_hess_[i].reserve(in_args_.size());
                for (size_t j : range(in_args_.size())) {
                    field_1 = in_args[i]->field();
                    field_2 = in_args[j]->field();
                    if (raw.prob_->contains(in_args[i]) &&
                        field_2 < field::num_prim &&
                        raw.prob_->contains(in_args[j])) {
                        /// @note order matches merit_data
                        /// h[i][j] = h[j][i] if i, j in the same field or field(i) < field(j)
                        /// otherwise only keep h[i][j] (empty)
                        if (func_.hess_sp_[i][j] == sparsity::unknown) {
                            goto BIND_EMPTY_HESS;
                        } else if (field_1 >= field_2) {
                            merit_hess_[i].push_back((*hessian)[field_1][field_2].insert(
                                raw.prob_->get_expr_start_tangent(in_args[i]),
                                raw.prob_->get_expr_start_tangent(in_args[j]),
                                in_args[i]->tdim(), in_args[j]->tdim(), func_.hess_sp_[i][j]));
                            continue;
                        }
                    }
                BIND_EMPTY_HESS:
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