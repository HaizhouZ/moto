#include <moto/ocp/impl/constr.hpp>

namespace moto {
namespace impl {
constr_approx_map::constr_approx_map(approx_storage &raw,
                                     sp_approx_map &&d)
    : constr_approx_map(raw.prob_->extract(raw.dual_[d.func_.field_], d.func_), raw, std::move(d)) {
}
constr_approx_map::constr_approx_map(vector_ref multiplier,
                                     approx_storage &raw,
                                     sp_approx_map &&d)
    : sp_approx_map(std::move(d)), merit_(&raw.merit_),
      multiplier_(multiplier) {
    auto f = &func_;
    const auto &in_args = f->in_args();
    if (f->order() >= approx_order::first) {
        map_merit_jac_from_raw(raw.jac_, vjp_);
    }
    if (f->order() >= approx_order::second) { // for hessian from vjp autodiff codegen
        in_args_.push_back(multiplier_);
    }
}
void constr_approx_map::map_merit_jac_from_raw(decltype(approx_storage::jac_) &raw, std::vector<row_vector_ref> &jac) {
    auto &in_args = func_.in_args();
    jac.clear();
    for (size_t i : range(in_args_.size())) {
        if (in_args[i]->field_ < field::num_prim) {
            jac.push_back(raw[in_args[i]->field_].segment(problem()->get_expr_start(in_args[i]), in_args[i]->dim_));
        } else { // useless
            static row_vector empty;
            jac.push_back(empty);
        }
    }
}
constr_approx_data::constr_approx_data(func &f) : f_(&f) {
    v_data_.resize(f.dim_);
    v_data_.setZero();
    jac_data_.reserve(f.in_args().size());
    for (auto &arg : f.in_args()) {
        if (arg->field_ < field::num_prim) {
            jac_data_.emplace_back(f.dim_, arg->dim_);
            jac_data_.back().setZero();
        } else { // useless
            static matrix empty;
            jac_data_.emplace_back(empty);
        };
    }
}
void constr::finalize_impl() {
    if (field_ == __undefined) {
        bool has_[3] = {false, false, false}; // x, u, y
        for (const auto &arg : in_args_) {
            if (arg->field_ <= __y)
                has_[arg->field_] = true;
        }
        // make this long enough so that people will not easily remove the const :D
        auto &field = const_cast<field_t &>(field_);
        if (field_hint_.is_eq == utils::optional_bool::Unset) {
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
                               name_, field::name(field_), arg->name_, arg->name_ + "_nxt");
                    substitute(arg, expr_lookup::get<sym>(arg->uid_ + 1));
                }
            }
        } catch (const std::exception &ex) {
            fmt::print("exception during substitution");
            throw;
        }
    }
    impl::func::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}

void constr::value_impl(sp_approx_map &data) {
    value(data);
    // compute contribution to merit function
    auto &d = static_cast<constr_approx_map &>(data);
    scalar_t res = d.multiplier_.dot(d.v_);
    // fmt::print("\t{}:\tv:{}\n", name_, d.v_.transpose());
    // #pragma omp critical
    // {
    // fmt::print("pre {}\n", *d.merit_);
    *d.merit_ += res;
    //     fmt::print("\t{}:\tv:{}\n", name_, d.multiplier_.dot(d.v_));
    //     fmt::print("after {}\n", *d.merit_);
    // }
    // fmt::print("\t{}:\tm:{}\n", name_, d.multiplier_.transpose());
} // namespace moto
void constr::jacobian_impl(sp_approx_map &data) {
    // compute jacobian first
    jacobian(data);
    // update multiplier - jacobian product
    auto &d = static_cast<constr_approx_map &>(data);
    for (size_t i = 0; i < d.in_arg_data().size(); i++) {
        if (d.vjp_[i].size() != 0) // skip if no jacobian for this input
            // fmt::print("{}\t{}:i\t{:.3}\n", i, name_, d.in_args_[i].transpose());
            d.vjp_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
        // fmt::print("{}\t{}:jac\n{:.3}\n", i, name_, d.jac_[i]);
    }
}
} // namespace impl
} // namespace moto
