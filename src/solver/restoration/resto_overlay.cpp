#include <moto/solver/restoration/resto_overlay.hpp>

#include <algorithm>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto::solver::restoration {
namespace {

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

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

void fill_sigma(const vector &ref, vector &sigma_sq, scalar_t eps) {
    sigma_sq.resizeLike(ref);
    if (ref.size() == 0) {
        return;
    }
    sigma_sq = ref.array().abs().max(eps).inverse().square().min(1.);
}

template <typename Overlay>
void copy_dual_slice(vector_ref dst, const node_data &outer, const Overlay &overlay) {
    const field_t source_field = overlay.source_field();
    if (source_field < __dyn || source_field >= field::num) {
        dst.setZero();
        return;
    }
    const auto &exprs = outer.problem().exprs(source_field);
    if (overlay.source_pos() >= exprs.size()) {
        throw std::runtime_error("restoration overlay source position out of range");
    }
    dst = outer.problem().extract(outer.dense().dual_[source_field], exprs[overlay.source_pos()]);
}

} // namespace

resto_prox_cost::resto_prox_cost(const std::string &name,
                                 const var_list &u_args,
                                 const var_list &y_args)
    : generic_cost(name, approx_order::second) {
    set_default_hess_sparsity(sparsity::diag);
    for (const sym &arg : u_args) {
        add_argument(arg);
    }
    for (const sym &arg : y_args) {
        add_argument(arg);
    }
}

func_approx_data_ptr_t resto_prox_cost::create_approx_data(sym_data &primal,
                                                           lag_data &raw,
                                                           shared_data &shared) const {
    return std::make_unique<approx_data>(primal, raw, shared, *this);
}

void resto_prox_cost::value_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t value = 0.;
    size_t arg_idx = 0;
    for (const sym &arg : in_args()) {
        const auto &x = d[arg];
        const auto *ref = arg.field() == __u ? &d.u_ref : &d.y_ref;
        const auto *sigma = arg.field() == __u ? &d.sigma_u_sq : &d.sigma_y_sq;
        if (x.size() == 0 || ref->size() == 0) {
            ++arg_idx;
            continue;
        }
        const vector delta = x - *ref;
        value += scalar_t(0.5) * sigma->dot(delta.cwiseProduct(delta));
        ++arg_idx;
    }
    d.v_(0) += value;
}

void resto_prox_cost::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t arg_idx = 0;
    for (const sym &arg : in_args()) {
        auto &grad = d.lag_jac_[arg_idx];
        if (grad.size() == 0) {
            ++arg_idx;
            continue;
        }
        const auto *ref = arg.field() == __u ? &d.u_ref : &d.y_ref;
        const auto *sigma = arg.field() == __u ? &d.sigma_u_sq : &d.sigma_y_sq;
        const vector delta = d[arg] - *ref;
        grad.array() += (sigma->array() * delta.array()).transpose();
        ++arg_idx;
    }
}

void resto_prox_cost::hessian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    for (size_t i = 0; i < in_args().size(); ++i) {
        const sym &arg = in_args()[i];
        if (arg.field() != __u && arg.field() != __y) {
            continue;
        }
        auto &block = d.lag_hess_[i][i];
        if (block.size() == 0) {
            continue;
        }
        const auto &sigma = arg.field() == __u ? d.sigma_u_sq : d.sigma_y_sq;
        block.diagonal().array() += sigma.array();
    }
}

resto_eq_elastic_constr::resto_eq_elastic_constr(const std::string &name,
                                                 const constr &source,
                                                 size_t source_pos,
                                                 scalar_t rho,
                                                 scalar_t lambda_reg)
    : soft_constr(name, approx_order::second, source->dim()),
      source_(source),
      source_pos_(source_pos),
      rho_(rho),
      lambda_reg_(lambda_reg) {
    field_hint().is_eq = true;
    field_hint().is_soft = true;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(dynamic_cast<const generic_func &>(*source).in_args());
}

void resto_eq_elastic_constr::setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {
    soft_constr::setup_workspace_data(data, ws_data);
    auto &d = data.as<approx_data>();
    d.ipm_cfg = &ws_data->as<solver::ipm_config>();
    d.rho = rho_;
    d.lambda_reg = lambda_reg_;
}

func_approx_data_ptr_t resto_eq_elastic_constr::create_approx_data(sym_data &primal,
                                                                   lag_data &raw,
                                                                   shared_data &shared) const {
    std::unique_ptr<soft_constr::approx_data> base_d(make_approx<soft_constr>(primal, raw, shared));
    return std::make_unique<approx_data>(std::move(*base_d));
}

void resto_eq_elastic_constr::value_impl(func_approx_data &data) const {
    forward_source_value(source_, data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    if (d.elastic.dim() == 0) {
        d.v_ = d.base_residual;
        return;
    }
    d.elastic.c_current = d.base_residual;
    d.v_ = d.base_residual - d.elastic.p + d.elastic.n;
}

void resto_eq_elastic_constr::jacobian_impl(func_approx_data &data) const {
    forward_source_jacobian(source_, data);
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || d.elastic.dim() == 0) {
        return;
    }
    compute_local_model(d.base_residual, d.multiplier_, d.elastic, d.rho, d.ipm_cfg->mu, d.lambda_reg);
    propagate_jacobian(d);
}

void resto_eq_elastic_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        forward_source_hessian(source_, data);
    }
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || d.elastic.dim() == 0) {
        return;
    }
    compute_local_model(d.base_residual, d.multiplier_, d.elastic, d.rho, d.ipm_cfg->mu, d.lambda_reg);
    propagate_hessian(d);
}

void resto_eq_elastic_constr::propagate_jacobian(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t arg_idx = 0;
    for (auto &jac : d.jac_) {
        if (jac.size() != 0) {
            d.lag_jac_corr_[arg_idx].noalias() += d.elastic.minv_bc.transpose() * jac;
        }
        ++arg_idx;
    }
}

void resto_eq_elastic_constr::propagate_hessian(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t outer_idx = 0;
    for (auto &outer : d.lag_hess_) {
        size_t inner_idx = 0;
        for (auto &inner : outer) {
            if (inner.size() != 0 && d.jac_[outer_idx].size() != 0 && d.jac_[inner_idx].size() != 0) {
                inner.noalias() += d.jac_[outer_idx].transpose() * d.elastic.minv_diag.asDiagonal() * d.jac_[inner_idx];
            }
            ++inner_idx;
        }
        ++outer_idx;
    }
}

void resto_eq_elastic_constr::propagate_res_stats(func_approx_data &) const {}

void resto_eq_elastic_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    d.elastic.resize(0, d.func_.dim());
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const auto init = initialize_elastic_pair(d.base_residual(i), d.rho, d.ipm_cfg->mu);
        d.elastic.p(i) = init.p;
        d.elastic.n(i) = init.n;
        d.elastic.nu_p(i) = init.z_p;
        d.elastic.nu_n(i) = init.z_n;
        d.multiplier_(i) = init.lambda;
    }
}

void resto_eq_elastic_constr::finalize_newton_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.elastic.dim() == 0) {
        return;
    }
    vector delta(d.func_.dim());
    delta.setZero();
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim) {
            delta.noalias() += d.jac_[arg_idx] * d.prim_step_[arg_idx];
        }
        ++arg_idx;
    }
    recover_local_step(delta, d.elastic, d.lambda_reg);
    d.d_multiplier_ = d.elastic.d_lambda;
}

void resto_eq_elastic_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    d.elastic.finalize_predictor_step(cfg->as<linesearch_config>(), cfg->as<solver::ipm_config::worker_type>());
}

void resto_eq_elastic_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    d.elastic.apply_affine_step(cfg->as<linesearch_config>());
    d.multiplier_.noalias() += cfg->as<linesearch_config>().dual_alpha_for_eq() * d.d_multiplier_;
}

void resto_eq_elastic_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    data.as<approx_data>().elastic.update_ls_bounds(cfg->as<linesearch_config>());
}

void resto_eq_elastic_constr::backup_trial_state(data_map_t &data) const {
    data.as<approx_data>().elastic.backup_trial_state();
}

void resto_eq_elastic_constr::restore_trial_state(data_map_t &data) const {
    data.as<approx_data>().elastic.restore_trial_state();
}

scalar_t resto_eq_elastic_constr::objective_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return d.rho * d.elastic.penalty_sum();
}

scalar_t resto_eq_elastic_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return d.rho * d.elastic.penalty_dir_deriv();
}

scalar_t resto_eq_elastic_constr::search_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    return d.ipm_cfg->mu * d.elastic.barrier_log_sum();
}

scalar_t resto_eq_elastic_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    return d.ipm_cfg->mu * d.elastic.barrier_dir_deriv();
}

scalar_t resto_eq_elastic_constr::local_stat_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return std::max(max_abs_or_zero(d.elastic.r_p), max_abs_or_zero(d.elastic.r_n));
}

scalar_t resto_eq_elastic_constr::local_comp_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return std::max(max_abs_or_zero(d.elastic.r_s_p), max_abs_or_zero(d.elastic.r_s_n));
}

resto_ineq_elastic_ipm_constr::resto_ineq_elastic_ipm_constr(const std::string &name,
                                                             const constr &source,
                                                             size_t source_pos,
                                                             scalar_t rho,
                                                             scalar_t lambda_reg)
    : ineq_constr(name, approx_order::second, source->dim()),
      source_(source),
      source_pos_(source_pos),
      rho_(rho),
      lambda_reg_(lambda_reg) {
    field_hint().is_eq = false;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(dynamic_cast<const generic_func &>(*source).in_args());
}

void resto_ineq_elastic_ipm_constr::setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {
    ineq_constr::setup_workspace_data(data, ws_data);
    auto &d = data.as<approx_data>();
    d.ipm_cfg = &ws_data->as<solver::ipm_config>();
    d.rho = rho_;
    d.lambda_reg = lambda_reg_;
}

func_approx_data_ptr_t resto_ineq_elastic_ipm_constr::create_approx_data(sym_data &primal,
                                                                         lag_data &raw,
                                                                         shared_data &shared) const {
    std::unique_ptr<ineq_constr::approx_data> base_d(make_approx<ineq_constr>(primal, raw, shared));
    return std::make_unique<approx_data>(std::move(*base_d));
}

void resto_ineq_elastic_ipm_constr::value_impl(func_approx_data &data) const {
    forward_source_value(source_, data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    if (d.elastic.dim() == 0) {
        d.v_ = d.base_residual;
        d.comp_.setZero();
        return;
    }
    d.elastic.g_current = d.base_residual;
    d.v_ = d.base_residual + d.elastic.t - d.elastic.p + d.elastic.n;
    d.comp_ = d.multiplier_.cwiseProduct(d.v_);
}

void resto_ineq_elastic_ipm_constr::jacobian_impl(func_approx_data &data) const {
    forward_source_jacobian(source_, data);
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || d.elastic.dim() == 0) {
        return;
    }
    compute_local_model(d.base_residual, d.multiplier_, d.elastic, d.rho, d.ipm_cfg->mu, d.lambda_reg);
    propagate_jacobian(d);
}

void resto_ineq_elastic_ipm_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        forward_source_hessian(source_, data);
    }
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || d.elastic.dim() == 0) {
        return;
    }
    compute_local_model(d.base_residual, d.multiplier_, d.elastic, d.rho, d.ipm_cfg->mu, d.lambda_reg);
    propagate_hessian(d);
}

void resto_ineq_elastic_ipm_constr::propagate_jacobian(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t arg_idx = 0;
    for (auto &jac : d.jac_) {
        if (jac.size() != 0) {
            d.lag_jac_corr_[arg_idx].noalias() += d.elastic.minv_bd.transpose() * jac;
        }
        ++arg_idx;
    }
}

void resto_ineq_elastic_ipm_constr::propagate_hessian(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t outer_idx = 0;
    for (auto &outer : d.lag_hess_) {
        size_t inner_idx = 0;
        for (auto &inner : outer) {
            if (inner.size() != 0 && d.jac_[outer_idx].size() != 0 && d.jac_[inner_idx].size() != 0) {
                inner.noalias() += d.jac_[outer_idx].transpose() * d.elastic.minv_diag.asDiagonal() * d.jac_[inner_idx];
            }
            ++inner_idx;
        }
        ++outer_idx;
    }
}

void resto_ineq_elastic_ipm_constr::propagate_res_stats(func_approx_data &) const {}

void resto_ineq_elastic_ipm_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    // The elastic IPM block is indexed by the wrapped inequality residual, not by
    // the active primal tangent dimensions. Keep its local state aligned with the
    // constraint dimension so local KKT assembly matches base_residual/multiplier_.
    d.elastic.resize(d.func_.dim(), 0);
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const auto init = initialize_elastic_ineq_scalar(d.base_residual(i), d.rho, d.ipm_cfg->mu);
        d.elastic.t(i) = init.t;
        d.elastic.p(i) = init.p;
        d.elastic.n(i) = init.n;
        d.elastic.nu_t(i) = init.nu_t;
        d.elastic.nu_p(i) = init.nu_p;
        d.elastic.nu_n(i) = init.nu_n;
        d.multiplier_(i) = init.lambda;
    }
}

void resto_ineq_elastic_ipm_constr::finalize_newton_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.elastic.dim() == 0) {
        return;
    }
    vector delta(d.func_.dim());
    delta.setZero();
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim) {
            delta.noalias() += d.jac_[arg_idx] * d.prim_step_[arg_idx];
        }
        ++arg_idx;
    }
    recover_local_step(delta, d.elastic, d.lambda_reg);
    d.d_multiplier_ = d.elastic.d_lambda;
}

void resto_ineq_elastic_ipm_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    d.elastic.finalize_predictor_step(cfg->as<linesearch_config>(), cfg->as<solver::ipm_config::worker_type>());
}

void resto_ineq_elastic_ipm_constr::apply_corrector_step(data_map_t &) const {}

void resto_ineq_elastic_ipm_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    const auto &ls = cfg->as<linesearch_config>();
    d.elastic.apply_affine_step(ls);
    d.multiplier_.noalias() += ls.dual_alpha_for_ineq() * d.d_multiplier_;
}

void resto_ineq_elastic_ipm_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    data.as<approx_data>().elastic.update_ls_bounds(cfg->as<linesearch_config>());
}

void resto_ineq_elastic_ipm_constr::backup_trial_state(data_map_t &data) const {
    data.as<approx_data>().elastic.backup_trial_state();
}

void resto_ineq_elastic_ipm_constr::restore_trial_state(data_map_t &data) const {
    data.as<approx_data>().elastic.restore_trial_state();
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return d.rho * d.elastic.penalty_sum();
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return d.rho * d.elastic.penalty_dir_deriv();
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    return d.ipm_cfg->mu * d.elastic.barrier_log_sum();
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    return d.ipm_cfg->mu * d.elastic.barrier_dir_deriv();
}

scalar_t resto_ineq_elastic_ipm_constr::local_stat_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return std::max({max_abs_or_zero(d.elastic.r_t),
                     max_abs_or_zero(d.elastic.r_p),
                     max_abs_or_zero(d.elastic.r_n)});
}

scalar_t resto_ineq_elastic_ipm_constr::local_comp_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    return std::max({max_abs_or_zero(d.elastic.r_s_t),
                     max_abs_or_zero(d.elastic.r_s_p),
                     max_abs_or_zero(d.elastic.r_s_n)});
}

ocp_ptr_t build_restoration_overlay_problem(const ocp_ptr_t &source_prob,
                                            const restoration_overlay_settings &settings) {
    ocp::active_status_config config;
    for (auto field : std::array{__cost, __eq_x, __eq_xu, __ineq_x, __ineq_xu, __eq_x_soft, __eq_xu_soft}) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            config.deactivate_list.emplace_back(*expr);
        }
    }

    auto resto_prob = std::static_pointer_cast<ocp>(source_prob->clone_base(config));

    var_list u_args;
    var_list y_args;
    for (const sym &arg : resto_prob->exprs(__u)) {
        u_args.emplace_back(arg);
    }
    for (const sym &arg : resto_prob->exprs(__y)) {
        y_args.emplace_back(arg);
    }
    if (!u_args.empty() || !y_args.empty()) {
        auto prox = cost(new resto_prox_cost("resto_prox", u_args, y_args));
        resto_prob->add(*prox);
    }

    for (auto field : std::array{__eq_x, __eq_xu}) {
        size_t source_pos = 0;
        for (const shared_expr &expr : source_prob->exprs(field)) {
            auto source = std::dynamic_pointer_cast<generic_constr>(expr);
            if (!source) {
                ++source_pos;
                continue;
            }
            auto overlay = constr(new resto_eq_elastic_constr(
                overlay_name(*source, "resto_eq"),
                source,
                source_pos,
                settings.rho_eq,
                settings.lambda_reg));
            resto_prob->add(*overlay);
            ++source_pos;
        }
    }

    for (auto field : std::array{__ineq_x, __ineq_xu}) {
        size_t source_pos = 0;
        for (const shared_expr &expr : source_prob->exprs(field)) {
            auto source = std::dynamic_pointer_cast<generic_constr>(expr);
            if (!source) {
                ++source_pos;
                continue;
            }
            auto overlay = constr(new resto_ineq_elastic_ipm_constr(
                overlay_name(*source, "resto_ineq"),
                source,
                source_pos,
                settings.rho_ineq,
                settings.lambda_reg));
            resto_prob->add(*overlay);
            ++source_pos;
        }
    }

    resto_prob->wait_until_ready();
    return resto_prob;
}

void seed_restoration_overlay_refs(node_data &resto, scalar_t prox_eps) {
    resto.for_each(__cost, [&](const resto_prox_cost &, resto_prox_cost::approx_data &d) {
        d.u_ref = d.primal_->value_[__u];
        d.y_ref = d.primal_->value_[__y];
        fill_sigma(d.u_ref, d.sigma_u_sq, prox_eps);
        fill_sigma(d.y_ref, d.sigma_y_sq, prox_eps);
    });
}

void sync_restoration_overlay_primal(node_data &outer, node_data &resto) {
    for (auto field : primal_fields) {
        resto.sym_val().value_[field] = outer.sym_val().value_[field];
    }
}

void sync_restoration_overlay_duals(node_data &outer, node_data &resto) {
    for (auto field : hard_constr_fields) {
        if (resto.dense().dual_[field].size() == 0 || outer.dense().dual_[field].size() == 0) {
            continue;
        }
        resto.dense().dual_[field] = outer.dense().dual_[field];
    }
    resto.for_each(__eq_x_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, overlay);
    });
    resto.for_each(__eq_xu_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, overlay);
    });
    resto.for_each(__ineq_x, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, overlay);
    });
    resto.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        copy_dual_slice(d.multiplier_, outer, overlay);
    });
}

void restore_outer_duals(array_type<vector, constr_fields> &dual,
                         const array_type<vector, constr_fields> &backup) {
    for (auto field : constr_fields) {
        dual[field] = backup[field];
    }
}

void commit_bound_state(vector_ref slack,
                        vector_ref multiplier,
                        const vector_const_ref &resto_slack,
                        const vector_const_ref &resto_multiplier,
                        scalar_t reset_threshold,
                        scalar_t reset_value) {
    slack = resto_slack;
    multiplier = resto_multiplier;
    maybe_reset_multiplier(multiplier, reset_threshold, reset_value);
}

bool should_reset_multiplier(const vector_const_ref &multiplier, scalar_t threshold) {
    return threshold <= 0.0 || (multiplier.size() > 0 && multiplier.cwiseAbs().maxCoeff() > threshold);
}

void maybe_reset_multiplier(vector_ref multiplier, scalar_t threshold, scalar_t reset_value) {
    if (should_reset_multiplier(multiplier, threshold)) {
        multiplier.setConstant(reset_value);
    }
}

void reset_equality_duals(array_type<vector, constr_fields> &dual, scalar_t threshold) {
    bool reset_any = false;
    for (auto field : std::array{__eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}) {
        if (should_reset_multiplier(dual[field], threshold)) {
            reset_any = true;
            break;
        }
    }
    if (!reset_any) {
        return;
    }
    for (auto field : std::array{__eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}) {
        dual[field].setZero();
    }
}

void reset_equality_duals(ns_riccati::ns_riccati_data &d, scalar_t threshold) {
    reset_equality_duals(d.dense_->dual_, threshold);
}

} // namespace moto::solver::restoration
