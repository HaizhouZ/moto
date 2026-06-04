#include <moto/solver/equality_init/eq_init_overlay.hpp>

#include <algorithm>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ineq_soft.hpp>

namespace moto::solver::equality_init {
namespace {

std::string overlay_name(const generic_func &source, std::string_view suffix) {
    return fmt::format("{}__{}", source.name(), suffix);
}

void copy_source_sparsity(generic_func &dst, const generic_func &src) {
    const auto &src_args = src.in_args();
    const auto &src_sp = src.jac_sparsity();
    for (size_t i = 0; i < src_args.size() && i < src_sp.size(); ++i) {
        dst.set_jac_sparsity(src_args[i], src_sp[i]);
    }
    dst.set_hess_sparsity(src.hess_sparsity());
}

template <typename Overlay>
void copy_dual_slice(vector_ref dst, const node_data &outer, const Overlay &overlay) {
    const auto &source_data = outer.data(overlay.source());
    dst = const_cast<func_approx_data &>(source_data).template as<generic_constr::approx_data>().multiplier_;
}

template <typename Overlay>
void commit_dual_slice(node_data &outer, const Overlay &overlay, const vector_const_ref &src) {
    auto &source_data = outer.data(overlay.source());
    source_data.template as<generic_constr::approx_data>().multiplier_ = src;
}

void sync_ineq_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    const auto &outer_exprs = outer.problem().exprs(field);
    const auto &overlay_exprs = overlay.problem().exprs(field);
    if (outer_exprs.size() != overlay_exprs.size()) {
        throw std::runtime_error(fmt::format("equality-init {} source/overlay size mismatch: {} vs {}",
                                             field::name(field), outer_exprs.size(), overlay_exprs.size()));
    }
    for (size_t i = 0; i < outer_exprs.size(); ++i) {
        auto &outer_ipm = outer.data(constr(outer_exprs[i])).as<solver::ipm_constr::approx_data>();
        auto &overlay_ipm = overlay.data(constr(overlay_exprs[i])).as<solver::ipm_constr::approx_data>();
        overlay_ipm.multiplier_ = outer_ipm.multiplier_;
        for (auto side : box_sides) {
            *overlay_ipm.box_side_[side] = *outer_ipm.box_side_[side];
        }
    }
}

void check_soft_overlay_prefix(const node_data &outer, const node_data &overlay, field_t field) {
    const auto &outer_exprs = outer.problem().exprs(field);
    const auto &overlay_exprs = overlay.problem().exprs(field);
    if (overlay_exprs.size() < outer_exprs.size()) {
        throw std::runtime_error(fmt::format("equality-init {} source/overlay prefix mismatch: {} vs {}",
                                             field::name(field), outer_exprs.size(), overlay_exprs.size()));
    }
}

void sync_soft_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    check_soft_overlay_prefix(outer, overlay, field);
    const auto &outer_exprs = outer.problem().exprs(field);
    const auto &overlay_exprs = overlay.problem().exprs(field);
    for (size_t i = 0; i < outer_exprs.size(); ++i) {
        auto &d = overlay.data(constr(overlay_exprs[i])).as<pmm_constr::approx_data>();
        d.multiplier_ = outer.problem().extract(outer.dense().dual_[field], outer_exprs[i]);
    }
}

void commit_soft_overlay_dual_field(node_data &outer, node_data &overlay, field_t field) {
    check_soft_overlay_prefix(outer, overlay, field);
    const auto &outer_exprs = outer.problem().exprs(field);
    const auto &overlay_exprs = overlay.problem().exprs(field);
    for (size_t i = 0; i < outer_exprs.size(); ++i) {
        auto &d = overlay.data(constr(overlay_exprs[i])).as<pmm_constr::approx_data>();
        auto dst = outer.problem().extract(outer.dense().dual_[field], outer_exprs[i]);
        dst = d.multiplier_;
    }
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
    copy_source_sparsity(*this, *source_func_);
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
    for (auto field : std::array{__eq_x, __eq_xu}) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            auto source = std::dynamic_pointer_cast<generic_constr>(expr);
            if (!source) {
                continue;
            }
            auto overlay = constr(new eq_init_pmm_constr(
                overlay_name(*source, "eq_init_pmm"),
                source,
                settings.rho_eq));
            overlay_prob->add(*overlay);
        }
    }

    overlay_prob->wait_until_ready();
    return overlay_prob;
}

void sync_equality_init_overlay_primal(node_data &outer, node_data &overlay) {
    for (auto field : primal_fields) {
        overlay.sym_val().value_[field] = outer.sym_val().value_[field];
    }
    overlay.sym_val().value_[__p] = outer.sym_val().value_[__p];
}

void sync_equality_init_overlay_duals(node_data &outer, node_data &overlay) {
    if (overlay.dense().dual_[__dyn].size() > 0 && outer.dense().dual_[__dyn].size() > 0) {
        overlay.dense().dual_[__dyn] = outer.dense().dual_[__dyn];
    }
    overlay.for_each(__eq_x, [&](const eq_init_pmm_constr &c, pmm_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, c);
    });
    overlay.for_each(__eq_xu, [&](const eq_init_pmm_constr &c, pmm_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, c);
    });
    sync_soft_overlay_dual_field(outer, overlay, __eq_x_soft);
    sync_soft_overlay_dual_field(outer, overlay, __eq_xu_soft);
    sync_ineq_overlay_dual_field(outer, overlay, __ineq_x);
    sync_ineq_overlay_dual_field(outer, overlay, __ineq_xu);
}

void commit_equality_init_overlay_duals(node_data &outer, node_data &overlay) {
    if (overlay.dense().dual_[__dyn].size() > 0 && outer.dense().dual_[__dyn].size() > 0) {
        outer.dense().dual_[__dyn] = overlay.dense().dual_[__dyn];
    }
    overlay.for_each(__eq_x, [&](const eq_init_pmm_constr &c, pmm_constr::approx_data &d) {
        commit_dual_slice(outer, c, d.multiplier_);
    });
    overlay.for_each(__eq_xu, [&](const eq_init_pmm_constr &c, pmm_constr::approx_data &d) {
        commit_dual_slice(outer, c, d.multiplier_);
    });
    commit_soft_overlay_dual_field(outer, overlay, __eq_x_soft);
    commit_soft_overlay_dual_field(outer, overlay, __eq_xu_soft);
}

} // namespace moto::solver::equality_init
