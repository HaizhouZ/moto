#include <moto/ocp/constr.hpp>
#include <iostream>
namespace moto {
constr_data::constr_data(approx_storage &raw,
                         sp_approx_map &&d,
                         constr_impl *f)
    : sp_approx_map(std::move(d)), merit_(&raw.cost_),
      multiplier_(raw.dual_[f->field_].segment(raw.prob_->get_expr_start(*f),
                                                f->dim_)) {
    const auto &in_args = f->in_args();
    for (size_t i = 0; i < in_args_.size(); i++) {
        vjp_.push_back(raw.jac_[in_args[i]->field_].segment(
            raw.prob_->get_expr_start(in_args[i]), in_args[i]->dim_));
    }
    if (f->order() >= approx_order::second) {
        in_args_.push_back(multiplier_);
    }
}

bool constr_impl::finalize_impl() {
    if (field_ == __undefined) {
        bool has_[3] = {false, false, false};
        for (const auto &arg : in_args_) {
            if (arg->field_ <= __y)
                has_[arg->field_] = true;
        }
        // make this long enough so that people will not easily remove the const :D
        auto &field = *const_cast<field_t *>(&field_);
        if (has_[__u] && !has_[__y])
            field = __eq_cstr_c;
        else if (has_[__x] && has_[__y])
            field = __dyn;
        else if (!has_[__u] && !has_[__x] && has_[__y])
            field = __eq_cstr_s;
        else
            throw std::runtime_error(fmt::format("unsupported constr type has_x: {}, has_u: {}, has_y: {}", has_[__x], has_[__u], has_[__y]));
    }
    if (field_ == __eq_cstr_s) {
        // do in_arg substitute
        try {
            for (auto &arg : in_args_) {
                if (arg->field_ == __x) {
#ifndef NDEBUG
                    fmt::print("replacing in arg {} of func {} with {}\n", arg->name_, name_, arg->name_ + "_nxt");
#endif
                    arg = sym(expr_index::get(arg->name_ + "_nxt"));
                }
            }
        } catch (const std::exception &ex) {
            fmt::print("substitute exception");
            throw;
        }
    }
    assert(field_ == __dyn || magic_enum::enum_name(field_).find(
                                  "cstr") != std::string::npos);
    return true;
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
        // fmt::print("{}\t{}:i\t{:.3}\n", i, name_, d.in_args_[i].transpose());
        d.vjp_[i].noalias() += d.multiplier_.transpose() * d.jac_[i];
        // fmt::print("{}\t{}:jac\n{:.3}\n", i, name_, d.jac_[i]);
    }
}
} // namespace moto
