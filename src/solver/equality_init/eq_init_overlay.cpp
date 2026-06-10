#include <moto/solver/equality_init/eq_init_overlay.hpp>

#include <algorithm>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/overlay_common.hpp>

namespace moto::solver::equality_init {
namespace {

void sync_ineq_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    solver::overlay::for_each_overlay_pair<solver::overlay::pair_match::same_size>(outer, overlay, field, "equality-init", [&](const constr &outer_expr, const constr &overlay_expr) {
        auto &outer_ipm = outer.data(outer_expr).as<solver::ipm_constr::approx_data>();
        auto &overlay_ipm = overlay.data(overlay_expr).as<solver::ipm_constr::approx_data>();
        overlay_ipm.multiplier_ = outer_ipm.multiplier_;
        for (auto side : box_sides) {
            *overlay_ipm.box_side_[side] = *outer_ipm.box_side_[side];
        }
    });
}

void sync_soft_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    solver::overlay::for_each_overlay_pair<solver::overlay::pair_match::prefix>(outer, overlay, field, "equality-init", [&](const constr &outer_expr, const constr &overlay_expr) {
        auto &d = overlay.data(overlay_expr).as<pmm_constr::approx_data>();
        d.multiplier_ = outer.problem().extract(outer.dense().dual_[field], outer_expr);
    });
}

void commit_soft_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    solver::overlay::for_each_overlay_pair<solver::overlay::pair_match::prefix>(outer, overlay, field, "equality-init", [&](const constr &outer_expr, const constr &overlay_expr) {
        auto &d = overlay.data(overlay_expr).as<pmm_constr::approx_data>();
        auto dst = outer.problem().extract(outer.dense().dual_[field], outer_expr);
        dst = d.multiplier_;
    });
}

} // namespace

eq_init_pmm_constr::eq_init_pmm_constr(const std::string &name,
                                       const constr &source,
                                       scalar_t rho)
    : pmm_constr(name, approx_order::second, source->dim()),
      source_(source),
      source_func_(dynamic_cast<const generic_func *>(source.get())) {
    if (source_func_ == nullptr) {
        throw std::runtime_error(fmt::format("eq_init_pmm_constr source {} is not a generic_func", source->name()));
    }
    this->rho = rho;
    field_hint_.is_eq = true;
    field_hint_.is_soft = true;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(source_func_->in_args());
    solver::overlay::copy_source_sparsity(*this, *source_func_);
}

void eq_init_pmm_constr::value_impl(func_approx_data &data) const {
    source_func_->value(data);
    auto &d = data.as<pmm_constr::approx_data>();
    solver::ineq_soft::ensure_initialized(*this, d);
    d.g_ = d.v_ - d.rho_ * d.multiplier_;
}

void eq_init_pmm_constr::jacobian_impl(func_approx_data &data) const {
    source_func_->jacobian(data);
    auto &d = data.as<pmm_constr::approx_data>();
    propagate_jacobian(d);
    propagate_hessian(d);
}

void eq_init_pmm_constr::hessian_impl(func_approx_data &data) const {
    if (source_func_->order() >= approx_order::second) {
        source_func_->hessian(data);
    }
}

ocp_ptr_t build_equality_init_overlay_problem(const ocp_ptr_t &source_prob,
                                              const equality_init_overlay_settings &settings) {
    ocp::active_status_config config;
    for (auto field : std::array{__eq_x, __eq_xu}) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            config.deactivate_list.emplace_back(*expr);
        }
    }

    auto overlay_prob = source_prob->clone(config);
    solver::overlay::add_constr_overlay_group(source_prob, overlay_prob, std::array{__eq_x, __eq_xu}, [&](const constr &source) {
        return constr(new eq_init_pmm_constr(solver::overlay::overlay_name(*source, "eq_init_pmm"),
                                             source,
                                             settings.rho_eq));
    });

    overlay_prob->wait_until_ready();
    return overlay_prob;
}

void sync_equality_init_overlay_primal(node_data &outer, node_data &overlay) {
    solver::overlay::copy_primal_and_params(outer, overlay);
}

void sync_equality_init_overlay_duals(node_data &outer, node_data &overlay) {
    solver::overlay::copy_dense_dual_if_present(outer, overlay, __dyn);
    solver::overlay::copy_source_multipliers<eq_init_pmm_constr>(outer, overlay, std::array{__eq_x, __eq_xu});
    sync_soft_overlay_dual_field(outer, overlay, __eq_x_soft);
    sync_soft_overlay_dual_field(outer, overlay, __eq_xu_soft);
    sync_ineq_overlay_dual_field(outer, overlay, __ineq_x);
    sync_ineq_overlay_dual_field(outer, overlay, __ineq_xu);
}

void commit_equality_init_overlay_duals(node_data &outer, node_data &overlay) {
    solver::overlay::copy_dense_dual_if_present(overlay, outer, __dyn);
    solver::overlay::commit_source_multipliers<eq_init_pmm_constr>(outer, overlay, std::array{__eq_x, __eq_xu});
    commit_soft_overlay_dual_field(outer, overlay, __eq_x_soft);
    commit_soft_overlay_dual_field(outer, overlay, __eq_xu_soft);
}

} // namespace moto::solver::equality_init
