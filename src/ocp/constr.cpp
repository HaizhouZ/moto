#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
generic_constr::approx_data::approx_data(func_approx_data &&d)
    : approx_data(d.merit_data_->prob_->extract(d.merit_data_->dual_[d.func_.field()], d.func_), *d.merit_data_, std::move(d)) {
}
generic_constr::approx_data::approx_data(vector_ref multiplier,
                                         merit_data &raw,
                                         func_approx_data &&d)
    : func_approx_data(std::move(d)), merit_(&raw.merit_),
      multiplier_(multiplier) {
    if (func_.order() >= approx_order::second) { // for hessian from vjp autodiff codegen
        in_args_.push_back(multiplier_);
    }
    if (in_field(func_.field(), merit_data::stored_constr_fields) && func_.field() != __dyn) {
        auto prob_ = merit_data_->prob_;
        size_t f_st = prob_->get_expr_start(func_);
        size_t arg_idx = 0;
        for (const sym &arg : func_.in_args()) {
            field_t f = arg.field();
            if (f < field::num_prim) {
                auto d = merit_data_->approx_[func_.field()].jac_[f].insert(
                    f_st, prob_->get_expr_start(arg), func_.dim(), arg.dim(), sparsity::dense);
                new (&jac_[arg_idx]) matrix_ref(d);
            }
            arg_idx++;
        }
    }
}
void generic_constr::approx_data::map_merit_jac_from_raw(decltype(merit_data::jac_) &raw, std::vector<row_vector_ref> &jac) {
    auto &in_args = func_.in_args();
    jac.clear();
    for (size_t i : range(in_args.size())) {
        if (in_args[i]->field() < field::num_prim) {
            jac.push_back(raw[in_args[i]->field()].segment(problem()->get_expr_start(in_args[i]), in_args[i]->dim()));
        } else { // useless
            static row_vector empty;
            jac.push_back(empty);
        }
    }
}

void generic_constr::finalize_impl() {
    if (field_ == __undefined) {
        bool has_[3] = {false, false, false}; // x, u, y
        for (const sym &arg : in_args_) {
            if (arg.field() <= __y)
                has_[arg.field()] = true;
        }
        // make this long enough so that people will not easily remove the const :D
        auto &_field = field_;
        if (field_hint_.is_eq == utils::optional_bool::Unset) {
            throw std::runtime_error(fmt::format("generic_constr {} eq/ineq hint unset, please set it using as_eq_ or as_ineq_", name_));
        }
        if (field_hint_.is_eq) {
            if (has_[__u] && !has_[__y])
                _field = field_hint_.is_soft ? __eq_xu_soft : __eq_xu;
            else if (has_[__x] && has_[__y] && !field_hint_.is_soft) // we dont assume x can be converted to y
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
    if (in_field(field_, std::array{__eq_x, __ineq_x, __eq_x_soft})) {
        // do in_arg substitute
        try {
            bool pure_x = arg_dim(__y) == 0;
            for (sym &arg : in_args_) {
                // here is a bit tricky, we substitute __x to __y if only __x exists in the in_args
                // but __y existing dont mean the constraint is solvable - probably it is not
                if (arg.field() == __x && pure_x) {
                    fmt::print("warning: substitution in generic_constr {} of type {}: inarg {} with {}\n",
                               name_, field::name(field_), arg.name(), arg.name() + "_nxt");
                    substitute(arg, arg.next());
                }
            }
        } catch (const std::exception &ex) {
            fmt::print("exception during substitution");
            throw;
        }
    }
    generic_func::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}

void generic_constr::value_impl(func_approx_data &data) const {
    base::value_impl(data);
    // compute contribution to merit function
    // auto &d = static_cast<approx_data &>(data);
    // scalar_t res = d.multiplier_.dot(d.v_);
    // fmt::print("\t{}:\tv:{}\n", name(), d.v_.transpose());
    // #pragma omp critical
    // {
    // fmt::print("pre {}\n", *d.merit_);
    // *d.merit_ += res;
    //     fmt::print("\t{}:\tv:{}\n", name(), d.multiplier_.dot(d.v_));
    //     fmt::print("after {}\n", *d.merit_);
    // }
    // fmt::print("\t{}:\tm:{}\n", name(), d.multiplier_.transpose());
} // namespace moto
void generic_constr::jacobian_impl(func_approx_data &data) const {
    // compute jacobian first
    auto &d = static_cast<approx_data &>(data);
    jacobian(data);
    // update multiplier - jacobian product
    // for (size_t i = 0; i < in_args_.size(); i++) {
    //     if (d.merit_jac_[i].size() != 0) { // skip if no jacobian for this input
    //         d.merit_jac_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
    //         // fmt::print("{}\t{}:i\t{:.3}\n", i, name(), d.in_args_[i].transpose());
    //         // fmt::print("{}\t{}:jac\n{:.3}\n", i, name(), d.jac_[i]);
    //     }
    // }
}
constr::constr(const std::string &name, approx_order order, size_t dim, field_t field)
    : func(generic_constr(name, order, dim, field)) {
} ///< create a generic constraint with name and dimension
constr::constr(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
               approx_order order, field_t field)
    : func(generic_constr(name, in_args, out, order, field)) {
    assert(out.size2() == 1 && "generic_constr output cols must be 1");
}
generic_constr *constr::operator->() const {
    return static_cast<generic_constr *>(base::operator->());
} ///< access the generic_func interface
} // namespace moto
