#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
void generic_cost::finalize_impl() {
    if (finalize_hint_.substitute_x_to_y) {
        for (sym &arg : in_args_) {
            switch (arg.field()) {
            case __x:
                // fmt::print("substitution in cost {}: inarg {} with {}\n",
                //            name_, arg.name(), arg.name() + "_nxt");
                substitute(arg, arg.next());
                break;
            case __u:
                throw std::runtime_error(fmt::format(
                    "cost {} can only be terminal state-only cost, but has input arguments of type {}",
                    name_, field::name(arg.field())));
            default:
                // do nothing
                break;
            }
        }
    }
    if (finalize_hint_.gauss_newton) {
        if (!gn_weight_) {
            throw std::runtime_error(fmt::format("cost {} gauss-newton weight not set. Did you provide a non-scalar output ?", name_));
        }
        add_argument(gn_weight_);
        skip_unused_arg_check_.insert(gn_weight_->uid());
        if (gen_.task_) {
            gen_.task_->gauss_newton = true;
            gen_.task_->weight_gn = gn_weight_;
        } else {
            /// @todo use gn_weight_ to scale the hessian
            hessian = [](func_approx_data &d) {
                // Gauss-Newton approximation: H ≈ J^T * J
                for (size_t i = 0; i < d.merit_hess_.size(); i++) {
                    for (size_t j = 0; j < d.merit_hess_[i].size(); j++) {
                        if (d.merit_hess_[i][j].size() > 0) {
                            d.merit_hess_[i][j] = d.jac_[i].transpose() * d.jac_[j];
                        }
                    }
                }
            };
        }
    }
    // fmt::print("field_hint for cost {} is {}\n", name_, finalize_hint_.substitute_x_to_y);
    // finalize the base class
    generic_func::finalize_impl();
    return;
}

generic_cost::generic_cost(const std::string &name, approx_order order)
    : generic_func(name, order, 1, __cost) {}

generic_cost::generic_cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out, approx_order order)
    : generic_func(name, in_args, out, order, __cost) {
    // assert(out.is_scalar() && "cost output must be a scalar");
    if (out.is_scalar()) {
    } else {
        finalize_hint_.gauss_newton = true;
    }
}

generic_cost* generic_cost::set_diag_hess() {
    set_default_hess_sparsity(sparsity::diag);
    return this;
}

generic_cost* generic_cost::as_terminal() {
    name_ += "_terminal";
    finalize_hint_.substitute_x_to_y = true;
    return this;
}

generic_cost* generic_cost::set_gauss_newton(const var &weight) {
    gn_weight_ = weight;
    finalize_hint_.gauss_newton = true;
    return this;
}

} // namespace moto