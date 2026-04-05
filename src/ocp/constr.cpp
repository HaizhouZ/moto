#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

#include <limits>
#include <unordered_set>

namespace moto {
void generic_constr::add_to_ocp_callback(ocp_base *prob) {
    lower_x_to_y_ = dynamic_cast<node_ocp *>(prob) == nullptr &&
                    dynamic_cast<edge_ocp *>(prob) == nullptr;
}

namespace {

struct normalized_box_bound {
    cs::SX sx;
};

struct affine_single_arg_model {
    vector offset;
    matrix jacobian;
};

vector sx_to_vector(const cs::SX &sx) {
    const Eigen::Index rows = static_cast<Eigen::Index>(sx.size1());
    const Eigen::Index cols = static_cast<Eigen::Index>(sx.size2());
    if (cols != 1) {
        throw std::runtime_error(fmt::format("expected column vector SX, got shape {}x{}", rows, cols));
    }
    vector out(rows);
    for (Eigen::Index i = 0; i < rows; ++i) {
        out(i) = cs::DM(sx(static_cast<casadi_int>(i))).scalar();
    }
    return out;
}

matrix sx_to_matrix(const cs::SX &sx) {
    const Eigen::Index rows = static_cast<Eigen::Index>(sx.size1());
    const Eigen::Index cols = static_cast<Eigen::Index>(sx.size2());
    matrix out(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            out(i, j) = cs::DM(sx(static_cast<casadi_int>(i), static_cast<casadi_int>(j))).scalar();
        }
    }
    return out;
}

normalized_box_bound normalize_box_bound(const generic_constr::box_bound_t &bound, casadi_int dim, std::string_view which) {
    return std::visit(
        [&](const auto &value) -> normalized_box_bound {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, scalar_t>) {
                return {.sx = cs::SX::ones(dim, 1) * value};
            } else if constexpr (std::is_same_v<T, vector>) {
                if (value.size() != dim) {
                    throw std::runtime_error(fmt::format("box {} bound has dim {}, expected {}", which, value.size(), dim));
                }
                std::vector<scalar_t> entries(value.data(), value.data() + value.size());
                return {.sx = cs::DM(entries)};
            } else {
                if (value.is_scalar()) {
                    return {.sx = cs::SX::repmat(value, dim, 1)};
                }
                if (value.numel() != dim) {
                    throw std::runtime_error(fmt::format("box {} bound has numel {}, expected {}", which, value.numel(), dim));
                }
                return {.sx = cs::SX::reshape(value, dim, 1)};
            }
        },
        bound);
}

bool scalar_is_pos_inf(const cs::SX &sx) {
    return sx.is_constant() && sx.scalar().is_inf();
}

bool scalar_is_neg_inf(const cs::SX &sx) {
    return sx.is_constant() && sx.scalar().is_minus_inf();
}

cs::SX build_box_residual(const cs::SX &out,
                          const normalized_box_bound &lb,
                          const normalized_box_bound &ub) {
    std::vector<cs::SX> rows;
    rows.reserve(static_cast<size_t>(2 * out.numel()));
    for (casadi_int i = 0; i < out.numel(); ++i) {
        const cs::SX out_i = out(i);
        const cs::SX lb_i = lb.sx(i);
        const cs::SX ub_i = ub.sx(i);
        if (!scalar_is_pos_inf(ub_i)) {
            rows.push_back(out_i - ub_i);
        }
        if (!scalar_is_neg_inf(lb_i)) {
            rows.push_back(lb_i - out_i);
        }
    }
    if (rows.empty()) {
        throw std::runtime_error("create_box received no finite lower or upper bounds");
    }
    return cs::SX::vertcat(rows);
}

void validate_residual_args(const var_inarg_list &args, const cs::SX &residual) {
    std::unordered_set<std::string> arg_names;
    arg_names.reserve(args.size());
    for (const sym &arg : args) {
        arg_names.insert(arg.name());
    }
    for (const cs::SX &s : cs::SX::symvar(residual)) {
        const std::string sym_name = s.name();
        bool covered = arg_names.contains(sym_name);
        if (!covered) {
            for (const sym &arg : args) {
                const std::string prefix = arg.name() + "_";
                if (sym_name.rfind(prefix, 0) == 0) {
                    covered = true;
                    break;
                }
            }
        }
        if (!covered) {
            throw std::runtime_error(fmt::format("box residual depends on symbolic input '{}' that is not listed in in_args", s.name()));
        }
    }
}

bool try_build_affine_single_arg_model(const var_inarg_list &args,
                                       const cs::SX &residual,
                                       affine_single_arg_model &model) {
    if (args.size() != 1) {
        return false;
    }
    const sym &arg = args.front().get();
    const cs::SX arg_sx = static_cast<const cs::SX &>(arg);
    const cs::SX jac_sx = cs::SX::jacobian(residual, arg_sx);
    if (!jac_sx.is_constant() || cs::SX::depends_on(jac_sx, arg_sx)) {
        return false;
    }
    const cs::SX offset_sx = cs::SX::substitute(residual, arg_sx, cs::SX::zeros(arg.dim(), 1));
    if (!offset_sx.is_constant()) {
        return false;
    }
    model.offset = sx_to_vector(offset_sx);
    model.jacobian = sx_to_matrix(jac_sx);
    return true;
}

constr make_affine_single_arg_box(const std::string &name,
                                  const sym &arg,
                                  affine_single_arg_model model,
                                  approx_order order,
                                  field_t field) {
    auto c = std::make_shared<generic_constr>(name, order, static_cast<size_t>(model.offset.size()), field);
    c->field_hint().is_eq = false;
    c->add_argument(arg);
    const matrix jacobian = std::move(model.jacobian);
    const vector offset = std::move(model.offset);
    c->value = [jacobian, offset](func_approx_data &d) {
        d.v_.noalias() = jacobian * d[0];
        d.v_ += offset;
    };
    c->jacobian = [jacobian](func_approx_data &d) {
        d.jac_[0] = jacobian;
    };
    c->hessian = [](func_approx_data &) {};
    return c;
}

} // namespace

constr generic_constr::create_box(const std::string &name,
                                  const var_inarg_list &args,
                                  const cs::SX &out,
                                  const box_bound_t &lb,
                                  const box_bound_t &ub,
                                  approx_order order,
                                  field_t field) {
    const casadi_int dim = out.numel();
    const cs::SX out_vec = cs::SX::reshape(out, dim, 1);
    const normalized_box_bound lb_norm = normalize_box_bound(lb, dim, "lower");
    const normalized_box_bound ub_norm = normalize_box_bound(ub, dim, "upper");
    const cs::SX residual = build_box_residual(out_vec, lb_norm, ub_norm);
    validate_residual_args(args, residual);

    affine_single_arg_model affine_model;
    if (try_build_affine_single_arg_model(args, residual, affine_model)) {
        return make_affine_single_arg_box(name, args.front().get(), std::move(affine_model), order, field);
    }

    auto c = std::make_shared<generic_constr>(name, args, residual, order, field);
    c->field_hint().is_eq = false;
    return c;
}

generic_constr::approx_data::approx_data(func_approx_data &&d)
    : approx_data(d.lag_data_->prob_->extract(d.lag_data_->dual_[d.func_.field()], d.func_), *d.lag_data_, std::move(d)) {
}
generic_constr::approx_data::approx_data(vector_ref multiplier,
                                         lag_data &raw,
                                         func_approx_data &&d)
    : func_approx_data(std::move(d)), lag_(&raw.lag_),
      multiplier_(multiplier) {
    if (func_.order() >= approx_order::second) { // for hessian from vjp autodiff codegen
        in_args_.push_back(multiplier_);
    }
    if (in_field(func_.field(), lag_data::stored_constr_fields) && func_.field() != __dyn) {
        auto prob_ = lag_data_->prob_;
        size_t f_st = prob_->get_expr_start(func_);
        size_t arg_idx = 0;
        for (const sym &arg : func_.in_args()) {
            field_t f = arg.field();
            if (f < field::num_prim && prob_->is_active(arg)) {
                auto d = lag_data_->approx_[func_.field()].jac_[f].insert(
                    f_st, prob_->get_expr_start_tangent(arg), func_.dim(), arg.tdim(), sparsity::dense);
                new (&jac_[arg_idx]) matrix_ref(d);
            }
            arg_idx++;
        }
    }
}
void generic_constr::approx_data::map_lag_jac_from_raw(decltype(lag_data::lag_jac_) &raw, std::vector<row_vector_ref> &jac) {
    auto &in_args = func_.in_args();
    jac.clear();
    for (size_t i = 0; i < in_args.size(); ++i) {
        if (in_args[i]->field() < field::num_prim && problem()->is_active(in_args[i])) {
            jac.push_back(problem()->extract_tangent(raw[in_args[i]->field()], in_args[i]));
        } else {
            static row_vector empty;
            jac.push_back(empty);
        }
    }
}

void generic_constr::finalize_impl() {
    if (field_ == __undefined) {
        bool has_[3] = {false, false, false};
        for (const sym &arg : in_args_) {
            if (arg.field() <= __y)
                has_[arg.field()] = true;
        }
        auto &_field = field_;
        if (field_hint_.is_eq == utils::optional_bool::Unset) {
            throw std::runtime_error(fmt::format("generic_constr {} eq/ineq hint unset, please set it using as_eq_ or cast_ineq_", name_));
        }
        if (field_hint_.is_eq) {
            if (has_[__u] && !has_[__y])
                _field = field_hint_.is_soft ? __eq_xu_soft : __eq_xu;
            else if (has_[__x] && has_[__y] && !field_hint_.is_soft)
                _field = __dyn;
            else if (!has_[__u] && (has_[__x] || has_[__y]))
                _field = field_hint_.is_soft ? __eq_x_soft : __eq_x;
            else
                throw std::runtime_error(fmt::format("unsupported eq generic_constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set _field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        } else {
            if (has_[__u] && !has_[__y])
                _field = __ineq_xu;
            else if (!has_[__u] && (has_[__x] || has_[__y]))
                _field = __ineq_x;
            else
                throw std::runtime_error(fmt::format("unsupported ineq generic_constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set _field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        }
    }
    // if ((lower_x_to_y_ || terminal_add_) &&
    //     in_field(field_, std::array{__eq_x, __ineq_x, __eq_x_soft})) {
    //     try {
    //         bool pure_x = true;
    //         for (const sym &arg : in_args_) {
    //             if (arg.field() == __y && in_field(arg.field(), primal_fields)) {
    //                 pure_x = false;
    //                 break;
    //             }
    //         }
    //         if (pure_x) {
    //             for (sym &arg : in_args_) {
    //                 if (arg.field() == __x) {
    //                     fmt::print("warning: substitution in generic_constr {} of type {}: inarg {} with {}\n",
    //                                name_, field::name(field_), arg.name(), arg.name() + "_nxt");
    //                     substitute(arg, arg.next());
    //                 }
    //             }
    //         }
    //     } catch (const std::exception &) {
    //         fmt::print("exception during substitution");
    //         throw;
    //     }
    // }
    generic_func::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}
} // namespace moto
