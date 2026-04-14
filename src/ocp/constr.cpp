#include <moto/ocp/constr.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
void generic_constr::add_to_ocp_callback(ocp_base *prob) {
    lower_x_to_y_ = dynamic_cast<node_ocp *>(prob) == nullptr &&
                    dynamic_cast<edge_ocp *>(prob) == nullptr;
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
        bool has_[3] = {false, false, false};
        for (const sym &arg : in_args_) {
            if (arg.field() <= __y)
                has_[arg.field()] = true;
        }
        auto &_field = field_;
        if (field_hint_.is_eq == utils::optional_bool::Unset) {
            throw std::runtime_error(fmt::format("generic_constr {} eq/ineq hint unset, please set field_hint().is_eq explicitly", name_));
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
    generic_func::finalize_impl();
    assert(field_ >= __dyn && field_ - __dyn < field::num_constr);
}
} // namespace moto
