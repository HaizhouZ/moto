#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
namespace {
bool is_inequality_relation(casadi_int op) {
    return op == casadi::OP_LT || op == casadi::OP_LE;
}

std::string relation_context(std::string_view context) {
    return context.empty() ? "constraint" : std::string(context);
}
} // namespace

cs::SX normalize_constraint_expression(const cs::SX &out,
                                        constraint_relation expected,
                                        std::string_view context) {
    std::vector<cs::SX> residuals;
    residuals.reserve(static_cast<size_t>(out.numel()));
    bool saw_relation = false;
    bool saw_plain = false;
    for (casadi_int i = 0; i < out.numel(); ++i) {
        const cs::SX entry = out(i);
        const casadi_int op = entry.op();
        const bool equality = op == casadi::OP_EQ;
        const bool inequality = is_inequality_relation(op);
        if (op == casadi::OP_NE) {
            throw std::invalid_argument(fmt::format(
                "{} does not support != constraints", relation_context(context)));
        }
        if (!equality && !inequality) {
            saw_plain = true;
            residuals.push_back(entry);
            continue;
        }
        saw_relation = true;
        if (expected == constraint_relation::equality && !equality) {
            throw std::invalid_argument(fmt::format(
                "{} is an equality constraint and only accepts == relations",
                relation_context(context)));
        }
        if (expected == constraint_relation::inequality && !inequality) {
            throw std::invalid_argument(fmt::format(
                "{} is an inequality constraint and only accepts <, <=, >, or >= relations",
                relation_context(context)));
        }
        residuals.push_back(entry.dep(0) - entry.dep(1));
    }
    if (saw_relation && saw_plain) {
        throw std::invalid_argument(fmt::format(
            "{} cannot mix relational and residual entries",
            relation_context(context)));
    }
    return saw_relation ? cs::SX::vertcat(residuals) : out;
}

constr generic_constr::create(const std::string &name, const cs::SX &out,
                              approx_order order, field_t field) {
    return create(name, var_inarg_list{}, out, order, field);
}

constr generic_constr::create(const std::string &name,
                              const var_inarg_list &args, const cs::SX &out,
                              approx_order order, field_t field) {
    return std::make_shared<generic_constr>(
        name, args,
        normalize_constraint_expression(out, constraint_relation::equality,
                                        name),
        order, field);
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
        bool has_[field::num_prim] = {};
        for (const sym &arg : in_args_) {
            if (arg.field() < field::num_prim)
                has_[arg.field()] = true;
        }
        auto &_field = field_;
        if (field_hint_.is_eq == utils::optional_bool::Unset) {
            throw std::runtime_error(fmt::format("generic_constr {} eq/ineq hint unset; use ineq_constr::create or pass an explicit constraint field", name_));
        }
        if (field_hint_.is_eq) {
            if (has_[__x] && has_[__y] && !field_hint_.is_soft)
                _field = __dyn;
            else if (has_[__l])
                throw std::runtime_error(fmt::format(
                    "constraint {} contains lifted variables; construct it with "
                    "moto.lifted.create() so its elimination group is explicit",
                    name_));
            else if (has_[__u] && !has_[__y])
                _field = field_hint_.is_soft ? __eq_xu_soft : __eq_xu;
            else if (!has_[__u] && (has_[__x] || has_[__y]))
                _field = field_hint_.is_soft ? __eq_x_soft : __eq_x;
            else
                throw std::runtime_error(fmt::format("unsupported eq generic_constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set _field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        } else {
            if (has_[__l])
                _field = __ineq_xu;
            else if (has_[__u] && !has_[__y])
                _field = __ineq_xu;
            else if (!has_[__u] && (has_[__x] || has_[__y]))
                _field = __ineq_x;
            else
                throw std::runtime_error(fmt::format("unsupported ineq generic_constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set _field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        }
    }
    generic_func::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}
} // namespace moto
