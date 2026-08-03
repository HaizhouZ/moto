#include <moto/core/external_function.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
cost generic_cost::make_tracking(const std::string &name,
                                 const var_inarg_list &args,
                                 const cs::SX &value,
                                 tracking_param weight_arg,
                                 tracking_param reference_arg) {
    const auto value_dim = static_cast<size_t>(value.numel());
    var value_symbol;
    for (const var &candidate : global_registry::infer_args(value)) {
        const cs::SX &candidate_sx = static_cast<const cs::SX &>(*candidate);
        if (candidate_sx.size1() == value.size1() &&
            candidate_sx.size2() == value.size2() &&
            cs::SX::is_equal(candidate_sx, value)) {
            value_symbol = candidate;
            break;
        }
    }
    const auto residual_dim = value_symbol
        ? static_cast<size_t>(value_symbol->tdim()) : value_dim;
    const auto resolve = [](const std::string &param_name,
                            size_t dim, tracking_param param) -> var {
        if (auto *symbol = std::get_if<var>(&param)) {
            if ((*symbol)->field() != __p)
                throw std::runtime_error(fmt::format(
                    "cost parameter {} must have parameter field, got {}",
                    param_name, field::name((*symbol)->field())));
            if ((*symbol)->dim() != dim)
                throw std::runtime_error(fmt::format(
                    "cost parameter {} has dim {}, expected {}", param_name,
                    (*symbol)->dim(), dim));
            return std::move(*symbol);
        }
        if (auto *scalar = std::get_if<scalar_t>(&param))
            return sym::params(param_name, dim, *scalar);
        return sym::params(param_name, dim, std::move(std::get<vector>(param)));
    };
    auto weight = resolve(name + "_weight", residual_dim, std::move(weight_arg));
    auto reference = resolve(name + "_reference", value_dim,
                             std::move(reference_arg));
    var_inarg_list all_args = args;
    all_args.emplace_back(*weight);
    all_args.emplace_back(*reference);
    const cs::SX value_vec = cs::SX::reshape(value, value_dim, 1);
    const cs::SX residual = value_symbol
        ? value_symbol->symbolic_difference(value_vec, reference)
        : value_vec - reference;
    const cs::SX output = residual_dim == 1
        ? scalar_t(0.5) * cs::SX::dot(residual, residual * weight)
        : residual;
    auto result = cost(new generic_cost(name, all_args, output));
    result->weight_ = std::move(weight);
    result->reference_ = std::move(reference);
    if (residual_dim > 1)
        result->gn_weight_ = result->weight_;
    return result;
}

void generic_cost::finalize_impl() {
    if (use_gauss_newton_) {
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
                for (size_t i = 0; i < d.lag_hess_.size(); i++) {
                    for (size_t j = 0; j < d.lag_hess_[i].size(); j++) {
                        if (d.lag_hess_[i][j].size() > 0) {
                            d.lag_hess_[i][j].noalias() += d.jac_[i].transpose() * d.jac_[j];
                        }
                    }
                }
            };
        }
    }
    // finalize the base class
    generic_func::finalize_impl();
    return;
}

generic_cost::generic_cost(const std::string &name, approx_order order)
    : generic_func(name, order, 1, __cost) {}

generic_cost::generic_cost(const std::string &name, const var_inarg_list &in_args, const cs::SX &out, approx_order order)
    : generic_func(name, in_args, out, order, __cost) {
    // assert(out.is_scalar() && "cost output must be a scalar");
    if (!out.is_scalar()) {
        use_gauss_newton_ = true;
    }
}

void generic_cost::substitute(const sym &arg, const sym &rhs) {
    generic_func::substitute(arg, rhs);
    if (bool(weight_) && *weight_ == arg)
        weight_ = expr_cast<sym>(rhs.handle());
    if (bool(reference_) && *reference_ == arg)
        reference_ = expr_cast<sym>(rhs.handle());
    if (bool(gn_weight_) && *gn_weight_ == arg)
        gn_weight_ = expr_cast<sym>(rhs.handle());
}

cost generic_cost::from_scalar(const std::string &name,
                               const var_inarg_list &args,
                               const cs::SX &value,
                               tracking_param weight,
                               tracking_param reference) {
    if (!value.is_scalar())
        throw std::runtime_error(fmt::format(
            "cost.from_scalar {} expected one value, got {}", name, value.numel()));
    return make_tracking(name, args, value, std::move(weight),
                         std::move(reference));
}

cost generic_cost::from_vector(const std::string &name,
                               const var_inarg_list &args,
                               const cs::SX &value,
                               tracking_param weight,
                               tracking_param reference) {
    if (value.numel() < 2)
        throw std::runtime_error(fmt::format(
            "cost.from_vector {} expected at least two values, got {}; use from_scalar for scalar output",
            name, value.numel()));
    return make_tracking(name, args, value, std::move(weight),
                         std::move(reference));
}

} // namespace moto
