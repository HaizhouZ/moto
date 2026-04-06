#include <moto/solver/equality_init/eq_init_overlay.hpp>

#include <algorithm>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto::solver::equality_init {
namespace {

template <typename Src>
void forward_source_value(const Src &source, func_approx_data &data) {
    dynamic_cast<const generic_func &>(*source).value(data);
}

template <typename Src>
void forward_source_jacobian(const Src &source, func_approx_data &data) {
    dynamic_cast<const generic_func &>(*source).jacobian(data);
}

template <typename Src>
void forward_source_hessian(const Src &source, func_approx_data &data) {
    dynamic_cast<const generic_func &>(*source).hessian(data);
}

std::string overlay_name(const generic_func &source, std::string_view suffix) {
    return fmt::format("{}__{}", source.name(), suffix);
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

} // namespace

eq_init_pmm_constr::eq_init_pmm_constr(const std::string &name,
                                       const constr &source,
                                       scalar_t rho)
    : pmm_constr(name, approx_order::second, source->dim()),
    source_(source) {
    this->rho = rho;
    field_hint().is_eq = true;
    field_hint().is_soft = true;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(dynamic_cast<const generic_func &>(*source).in_args());
}

void eq_init_pmm_constr::value_impl(func_approx_data &data) const {
    forward_source_value(source_, data);
    auto &d = data.as<pmm_constr::approx_data>();
    d.g_ = d.v_ - d.rho_ * d.multiplier_;
}

void eq_init_pmm_constr::jacobian_impl(func_approx_data &data) const {
    forward_source_jacobian(source_, data);
    pmm_constr::propagate_jacobian(data);
    pmm_constr::propagate_hessian(data);
}

void eq_init_pmm_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        forward_source_hessian(source_, data);
    }
    pmm_constr::propagate_hessian(data);
}

ocp_ptr_t build_equality_init_overlay_problem(const ocp_ptr_t &source_prob,
                                              const equality_init_overlay_settings &settings) {
    ocp::active_status_config config;
    for (auto field : std::array{__eq_x, __eq_xu}) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            config.deactivate_list.emplace_back(*expr);
        }
    }

    auto overlay_prob = std::static_pointer_cast<ocp>(source_prob->clone_base(config));
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
    size_t pos_eq_x_soft = 0;
    overlay.for_each(__eq_x_soft, [&](const generic_constr &c, func_approx_data &ad) {
        if (dynamic_cast<const eq_init_pmm_constr *>(&c) != nullptr) {
            return;
        }
        auto &d = ad.as<pmm_constr::approx_data>();
        const auto &exprs = outer.problem().exprs(__eq_x_soft);
        if (pos_eq_x_soft >= exprs.size()) {
            throw std::runtime_error("equality-init soft eq_x source position out of range");
        }
        d.multiplier_ = outer.problem().extract(outer.dense().dual_[__eq_x_soft], exprs[pos_eq_x_soft++]);
    });
    size_t pos_eq_xu_soft = 0;
    overlay.for_each(__eq_xu_soft, [&](const generic_constr &c, func_approx_data &ad) {
        if (dynamic_cast<const eq_init_pmm_constr *>(&c) != nullptr) {
            return;
        }
        auto &d = ad.as<pmm_constr::approx_data>();
        const auto &exprs = outer.problem().exprs(__eq_xu_soft);
        if (pos_eq_xu_soft >= exprs.size()) {
            throw std::runtime_error("equality-init soft eq_xu source position out of range");
        }
        d.multiplier_ = outer.problem().extract(outer.dense().dual_[__eq_xu_soft], exprs[pos_eq_xu_soft++]);
    });
    size_t pos_ineq_x = 0;
    outer.for_each(__ineq_x, [&](const generic_constr &, func_approx_data &outer_ad) {
        auto &outer_ipm = outer_ad.as<solver::ipm_constr::approx_data>();
        size_t overlay_pos = 0;
        bool found = false;
        overlay.for_each(__ineq_x, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &overlay_ad) {
            if (overlay_pos++ != pos_ineq_x) {
                return;
            }
            overlay_ad.multiplier_ = outer_ipm.multiplier_;
            overlay_ad.multiplier_backup_ = outer_ipm.multiplier_backup_;
            overlay_ad.slack_ = outer_ipm.slack_;
            overlay_ad.slack_backup_ = outer_ipm.slack_backup_;
            found = true;
        });
        if (!found) {
            throw std::runtime_error("equality-init ineq_x source position out of range");
        }
        ++pos_ineq_x;
    });
    size_t pos_ineq_xu = 0;
    outer.for_each(__ineq_xu, [&](const generic_constr &, func_approx_data &outer_ad) {
        auto &outer_ipm = outer_ad.as<solver::ipm_constr::approx_data>();
        size_t overlay_pos = 0;
        bool found = false;
        overlay.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &overlay_ad) {
            if (overlay_pos++ != pos_ineq_xu) {
                return;
            }
            overlay_ad.multiplier_ = outer_ipm.multiplier_;
            overlay_ad.multiplier_backup_ = outer_ipm.multiplier_backup_;
            overlay_ad.slack_ = outer_ipm.slack_;
            overlay_ad.slack_backup_ = outer_ipm.slack_backup_;
            found = true;
        });
        if (!found) {
            throw std::runtime_error("equality-init ineq_xu source position out of range");
        }
        ++pos_ineq_xu;
    });
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
    size_t pos_eq_x_soft = 0;
    overlay.for_each(__eq_x_soft, [&](const generic_constr &c, func_approx_data &ad) {
        if (dynamic_cast<const eq_init_pmm_constr *>(&c) != nullptr) {
            return;
        }
        auto &d = ad.as<pmm_constr::approx_data>();
        auto &exprs = outer.problem().exprs(__eq_x_soft);
        if (pos_eq_x_soft >= exprs.size()) {
            throw std::runtime_error("equality-init soft eq_x source position out of range");
        }
        auto dst = outer.problem().extract(outer.dense().dual_[__eq_x_soft], exprs[pos_eq_x_soft++]);
        dst = d.multiplier_;
    });
    size_t pos_eq_xu_soft = 0;
    overlay.for_each(__eq_xu_soft, [&](const generic_constr &c, func_approx_data &ad) {
        if (dynamic_cast<const eq_init_pmm_constr *>(&c) != nullptr) {
            return;
        }
        auto &d = ad.as<pmm_constr::approx_data>();
        auto &exprs = outer.problem().exprs(__eq_xu_soft);
        if (pos_eq_xu_soft >= exprs.size()) {
            throw std::runtime_error("equality-init soft eq_xu source position out of range");
        }
        auto dst = outer.problem().extract(outer.dense().dual_[__eq_xu_soft], exprs[pos_eq_xu_soft++]);
        dst = d.multiplier_;
    });
}

} // namespace moto::solver::equality_init
