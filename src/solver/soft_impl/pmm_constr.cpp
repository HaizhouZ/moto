#include <moto/solver/soft_constr/pmm_constr.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ineq_soft.hpp>

namespace moto {
namespace solver {

using pmm_data = pmm_constr::approx_data;

pmm_constr::approx_data::approx_data(base::approx_data &&rhs, scalar_t rho)
    : base::approx_data(std::move(rhs)), rho_(rho) {
    g_.resize(func_.dim());
    g_.setZero();
    jac_step_.resize(func_.dim());
    jac_step_.setZero();
    multiplier_backup_.resize(func_.dim());
    multiplier_backup_.setZero();
}

void pmm_constr::initialize(data_map_t &data) const {
    auto &d = data.as<pmm_data>();
    d.multiplier_.setZero();
}

void pmm_constr::value_impl(func_approx_data &data) const {
    base::value_impl(data);
    auto &d = data.as<pmm_data>();
    solver::ineq_soft::ensure_initialized(*this, d);
    d.g_ = d.v_ - d.rho_ * d.multiplier_;  // raw C(x) = h; v_ is the primal residual used by inf_prim_res and merit
}

void pmm_constr::jacobian_impl(func_approx_data &data) const {
    base::jacobian_impl(data);
    propagate_jacobian(data);
    propagate_hessian(data);
}

void pmm_constr::propagate_jacobian(func_approx_data &data) const {
    auto &d = data.as<pmm_data>();
    soft_constr::propagate_jacobian(d, d.g_, scalar_t(1.) / d.rho_);
}

void pmm_constr::propagate_hessian(func_approx_data &data) const {
    auto &d = data.as<pmm_data>();
    soft_constr::propagate_hessian(d, scalar_t(1. / d.rho_));
}

void pmm_constr::finalize_newton_step(data_map_t &data) const {
    auto &d = data.as<pmm_data>();
    if (!d.runtime_bound_) {
        throw std::runtime_error(fmt::format("pmm_constr {} finalize_newton_step without runtime binding", name()));
    }
    if (!d.initialized_) {
        throw std::runtime_error(fmt::format("pmm_constr {} finalize_newton_step before initialization", name()));
    }
    if (d.d_multiplier_.size() != d.g_.size()) {
        throw std::runtime_error(fmt::format("pmm_constr {} d_multiplier size mismatch: {} vs g {}",
                                             name(), d.d_multiplier_.size(), d.g_.size()));
    }
    if (d.prim_step_.size() != static_cast<size_t>(d.func_.in_args().size())) {
        throw std::runtime_error(fmt::format("pmm_constr {} prim_step size mismatch: {} vs in_args {}",
                                             name(), d.prim_step_.size(), d.func_.in_args().size()));
    }
    // From row 2 of KKT: J*du - rho*dlam = -h  =>  dlam = (J*du + h) / rho
    d.d_multiplier_.noalias() = d.g_;
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim && d.has_jacobian_block(arg_idx)) {
            d.jac_step_.noalias() = d.jac_[arg_idx] * d.prim_step_[arg_idx];
            d.d_multiplier_.noalias() += d.jac_step_;
        }
        arg_idx++;
    }
    d.d_multiplier_ /= d.rho_;
}

void pmm_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<pmm_data>();
    auto &ls_cfg = cfg->as<solver::linesearch_config>();
    const scalar_t alpha = ls_cfg.dual_alpha_for_eq();
    d.multiplier_.noalias() += alpha * d.d_multiplier_;
}

void pmm_constr::backup_trial_state(data_map_t &data) const {
    auto &d = data.as<pmm_data>();
    d.multiplier_backup_ = d.multiplier_;
}

void pmm_constr::restore_trial_state(data_map_t &data) const {
    auto &d = data.as<pmm_data>();
    d.multiplier_ = d.multiplier_backup_;
}

} // namespace solver
} // namespace moto
