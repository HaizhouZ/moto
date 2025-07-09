#include <moto/ocp/constr.hpp>

namespace moto {
constr_data::constr_data(approx_storage &raw,
                         sp_approx_map &&d,
                         constr_impl *f)
    : sp_approx_map(std::move(d)), merit_(&raw.cost_),
      multiplier_(raw.dual_[f->field_].segment(raw.prob_->get_expr_start(*f),
                                               f->dim_)) {
    const auto &in_args = f->in_args();
    multiplier_.setConstant(1.);
    for (size_t i = 0; i < in_args_.size(); i++) {
        if (in_args[i]->field_ < field::num_prim)
            vjp_.push_back(raw.jac_[in_args[i]->field_].segment(
                raw.prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
        else {
            static row_vector empty;
            vjp_.push_back(empty); // no jacobian for this field
        }
    }
    if (f->order() >= approx_order::second) {
        in_args_.push_back(multiplier_);
    }
}
void constr_impl::finalize_impl() {
    if (field_ == __undefined) {
        bool has_[3] = {false, false, false}; // x, u, y
        for (const auto &arg : in_args_) {
            if (arg->field_ <= __y)
                has_[arg->field_] = true;
        }
        // make this long enough so that people will not easily remove the const :D
        auto &field = const_cast<field_t &>(field_);
        if (field_hint_.is_eq == utils::Unset) {
            throw std::runtime_error(fmt::format("constr {} eq/ineq hint unset, please set it using as_eq() or as_ineq()", name_));
        }
        if (field_hint_.is_eq) {
            if (has_[__u] && !has_[__y])
                field = field_hint_.is_soft ? __eq_xu_soft : __eq_xu;
            else if (has_[__x] && has_[__y] && !field_hint_.is_soft) // we dont assume x can be converted to y
                field = __dyn;
            else if (!has_[__u] && (has_[__x] ^ has_[__y]))
                field = field_hint_.is_soft ? __eq_x_soft : __eq_x;
            else
                throw std::runtime_error(fmt::format("unsupported eq constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        } else {
            if (has_[__u] && !has_[__y] && !field_hint_.is_soft)
                field = __ineq_xu;
            else if (!has_[__u] && (has_[__x] ^ has_[__y]) && !field_hint_.is_soft)
                field = __ineq_x;
            else
                throw std::runtime_error(fmt::format("unsupported ineq constr \"{}\" type has_x: {}, has_u: {}, has_y: {}, soft: {}. Did you set field or hints?",
                                                     name_, has_[__x], has_[__u], has_[__y], field_hint_.is_soft));
        }
    }
    if (field_ == __eq_x || field_ == __ineq_x || field_ == __eq_x_soft) {
        // do in_arg substitute
        try {
            for (auto &arg : in_args_) {
                if (arg->field_ == __x) {
                    fmt::print("substitution in constr {} of type {}: inarg {} with {}\n",
                               name_, magic_enum::enum_name(field_), arg->name_, arg->name_ + "_nxt");
                    substitute(arg, expr_lookup::get<sym>(arg->name_ + "_nxt"));
                }
            }
        } catch (const std::exception &ex) {
            fmt::print("exception during substitution");
            throw;
        }
    }
    func_impl::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}

void constr_impl::value_impl(sp_approx_map &data) {
    value(data);
    // compute contribution to merit function
    auto &d = static_cast<constr_data &>(data);
    // fmt::print("\t{}:\tv:{}\n", name_, d.v_.transpose());
    // #pragma omp critical
    // {
    // fmt::print("pre {}\n", *d.merit_);
    *d.merit_ += d.multiplier_.dot(d.v_);
    //     fmt::print("\t{}:\tv:{}\n", name_, d.multiplier_.dot(d.v_));
    //     fmt::print("after {}\n", *d.merit_);
    // }
    // fmt::print("\t{}:\tm:{}\n", name_, d.multiplier_.transpose());
} // namespace moto
void constr_impl::jacobian_impl(sp_approx_map &data) {
    // compute jacobian first
    jacobian(data);
    // update multiplier - jacobian product
    auto &d = static_cast<constr_data &>(data);
    for (size_t i = 0; i < d.in_args().size(); i++) {
        if (d.vjp_[i].cols()) // skip if no jacobian for this input
            // fmt::print("{}\t{}:i\t{:.3}\n", i, name_, d.in_args_[i].transpose());
            d.vjp_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
        // fmt::print("{}\t{}:jac\n{:.3}\n", i, name_, d.jac_[i]);
    }
}
} // namespace moto
