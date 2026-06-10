#include <moto/solver/restoration/resto_overlay.hpp>

#include <algorithm>
#include <cmath>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <moto/solver/overlay_common.hpp>
#include <string_view>
#include <tuple>
#include <utility>

namespace moto::solver::restoration {

namespace {
constexpr std::array k_pair_slots{detail::slot_p, detail::slot_n};
constexpr std::array<scalar_t, 2> k_pair_signs{scalar_t(-1.), scalar_t(1.)};
constexpr std::array k_triplet_slots{detail::slot_t, detail::slot_p, detail::slot_n};

constexpr scalar_t side_jac_sign(box_side::side_t side) {
    return side == box_side::ub ? scalar_t(1.) : scalar_t(-1.);
}

size_t local_state_dim(const detail::eq_local_state &state) {
    return state.ns + state.nc;
}

size_t local_state_dim(const detail::ineq_local_state &state) {
    return state.nx + state.nu;
}

template <typename LocalState>
void require_local_state_initialized(const LocalState &state, Eigen::Index expected_dim, std::string_view where) {
    const auto dim = static_cast<Eigen::Index>(local_state_dim(state));
    if (dim != expected_dim) {
        throw std::runtime_error(fmt::format("{} requires initialized local restoration state of size {}, got {}",
                                             where, expected_dim, dim));
    }
}

template <typename ApproxData>
scalar_t rho_value(const ApproxData &d, std::string_view where) {
    if (d.rho == nullptr) {
        throw std::runtime_error(fmt::format("{} requires restoration rho in workspace data", where));
    }
    return *d.rho;
}

template <typename Elastic, typename Fn>
void for_each_active_ineq_side(Elastic &&elastic,
                               const ineq_constr::box_spec &box,
                               Fn &&fn) {
    for (auto side : box_sides) if (box.has_side[side]) {
        fn(side, elastic.side[side], box.present_mask[side]);
    }
}

template <typename Elastic, typename Fn>
void for_each_active_ineq_side_slot(Elastic &&elastic,
                                    const ineq_constr::box_spec &box,
                                    Fn &&fn) {
    for_each_active_ineq_side(std::forward<Elastic>(elastic), box, [&](auto side, auto &side_state, const auto &mask) {
        for (auto slot : k_triplet_slots) {
            fn(side, slot, side_state, mask);
        }
    });
}

void fill_sigma_tangent(const sym &arg,
                        vector_ref ref,
                        vector_ref sigma_sq,
                        scalar_t eps,
                        scalar_t weight) {
    if (sigma_sq.size() == 0) {
        return;
    }
    sigma_sq = weight * ref.array().abs().max(eps).inverse().square().min(1.);
}

vector compute_tangent_delta(const sym &arg, vector_ref x, vector_ref ref) {
    vector delta(arg.tdim());
    arg.difference(x, ref, delta);
    return delta;
}

template <typename Overlay>
void copy_ineq_side_init(ineq_constr::box_side_array<vector> &slack_dst,
                         ineq_constr::box_side_array<vector> &dual_dst,
                         const node_data &outer,
                         const Overlay &overlay) {
    const auto *ipm_source = dynamic_cast<const solver::ipm_constr *>(overlay.source().get());
    if (ipm_source == nullptr) {
        throw std::runtime_error("restoration inequality overlay requires boxed ipm source");
    }
    const auto &source_data = outer.data(overlay.source());
    const auto &ipm = const_cast<func_approx_data &>(source_data).template as<solver::ipm_constr::ipm_data>();
    const auto *box = ipm_source->box_info();
    if (box == nullptr) {
        throw std::runtime_error("boxed ipm missing box_info in copy_ineq_side_init");
    }
    const auto n = static_cast<Eigen::Index>(box->base_dim);
    for (auto side : box_sides) if (box->has_side[side]) {
        slack_dst[side].resize(n);
        dual_dst[side].resize(n);
        slack_dst[side] = ipm.box_side_[side]->slack;
        dual_dst[side] = ipm.box_side_[side]->multiplier;
    }
}

void update_ineq_side_residuals(resto_ineq_elastic_ipm_constr::approx_data &d) {
    const auto &box = d.require_box_spec("update_ineq_side_residuals");
    d.elastic.present_mask = box.present_mask;
    ineq_constr::box_side_array<vector> bound_eval;
    for_each_active_ineq_side(d.elastic, box, [&](auto side, auto &side_state, const auto &mask) {
        if (box.bound_source[side] == ineq_constr::box_bound_source::constant) {
            bound_eval[side] = d.box_const_[side];
        } else if (box.bound_source[side] == ineq_constr::box_bound_source::in_arg) {
            bound_eval[side] = d[static_cast<const sym &>(box.bound_var[side])];
        } else {
            bound_eval[side].setZero(d.base_residual.size());
        }
        side_state.residual =
            mask.select(side_jac_sign(side) * (d.base_residual.array() - bound_eval[side].array()), scalar_t(0))
                .matrix();
    });
}

void sync_ineq_overlay_views(resto_ineq_elastic_ipm_constr::approx_data &d) {
    const auto &box = d.require_box_spec("sync_ineq_overlay_views");
    d.v_ = d.elastic.primal_view;
    d.multiplier_.setZero();
    d.d_multiplier_.setZero();
    d.comp_.setZero();
    for_each_active_ineq_side(d.elastic, box, [&](auto side, const auto &side_state, const auto &mask) {
        const scalar_t dual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
        d.multiplier_.array() +=
            dual_sign * mask.select(side_state.dual[detail::slot_t].array(), scalar_t(0));
        d.d_multiplier_.array() +=
            dual_sign * mask.select(side_state.d_dual[detail::slot_t].array(), scalar_t(0));
    });
}

using ineq_corrector_view = detail::elastic_side_array<detail::elastic_triplet_array<vector>>;

void refresh_ineq_local_model(resto_ineq_elastic_ipm_constr::approx_data &d,
                              const ineq_constr::box_spec &box,
                              std::string_view where,
                              scalar_t mu,
                              const ineq_corrector_view *corrector = nullptr) {
    update_ineq_side_residuals(d);
    resto_ineq_elastic_ipm_constr::compute_local_model(d.elastic, box, rho_value(d, where), mu, corrector);
    sync_ineq_overlay_views(d);
}

void finalize_predictor_pairs_like_ipm(resto_eq_elastic_constr::approx_data &d,
                                       workspace_data *cfg) {
    auto &worker = cfg->as<solver::ipm_config::worker_type>();
    auto &ls = cfg->as<linesearch_config>();
    assert(d.ipm_cfg != nullptr);
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    const scalar_t alpha_primal = ls.alpha_primal;
    const scalar_t alpha_dual = ls.alpha_dual;
    for (auto slot : k_pair_slots) {
        const auto &value = d.elastic.value[slot];
        if (value.size() == 0) {
            continue;
        }
        worker.n_ipm_cstr += static_cast<size_t>(value.size());
        worker.prev_aff_comp += d.elastic.dual[slot].dot(value);
        worker.post_aff_comp +=
            (d.elastic.dual[slot] + alpha_dual * d.elastic.d_dual[slot])
                .dot(value + alpha_primal * d.elastic.d_value[slot]);
    }
    for (auto slot : k_pair_slots) {
        d.elastic.corrector[slot].array() =
            ls.alpha_dual * d.elastic.d_dual[slot].array() * ls.alpha_primal *
            d.elastic.d_value[slot].array();
    }
}

void apply_corrector_pairs_like_ipm(resto_eq_elastic_constr::approx_data &d) {
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("apply_corrector_pairs_like_ipm requires ipm_cfg");
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "apply_corrector_pairs_like_ipm");
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        for (auto slot : k_pair_slots) {
            d.elastic.corrector[slot].setZero();
        }
    }
}

void finalize_pair_newton_step(resto_eq_elastic_constr::approx_data &d) {
    require_local_state_initialized(d.elastic, d.func_.dim(), "finalize_pair_newton_step");
    d.jac_step.setZero(d.func_.dim());
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim && d.has_jacobian_block(arg_idx)) {
            d.jac_step_tmp.noalias() = d.jac_[arg_idx] * d.prim_step_[arg_idx];
            d.jac_step.noalias() += d.jac_step_tmp;
        }
        ++arg_idx;
    }
    const scalar_t eps = 1e-16;
    d.elastic.d_multiplier = d.elastic.schur_inv_diag.array() * (d.jac_step.array() + d.elastic.condensed_rhs.array());
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        const auto scale = d.elastic.value[slot].array() / d.elastic.dual[slot].array().max(eps);
        d.elastic.d_value[slot] = -sign * scale * d.elastic.d_multiplier.array() - d.elastic.backsub_rhs[slot].array();
        d.elastic.d_dual[slot] = d.elastic.r_stat[slot].array() + sign * d.elastic.d_multiplier.array();
    }
    d.d_multiplier_ = d.elastic.d_multiplier;
}

void apply_affine_pairs_like_ipm(resto_eq_elastic_constr::approx_data &d,
                                 workspace_data *cfg,
                                 scalar_t multiplier_dual_alpha,
                                 scalar_t pair_dual_alpha) {
    auto &ls = cfg->as<linesearch_config>();
    assert(d.ipm_cfg != nullptr);
    assert(!d.ipm_cfg->ipm_computing_affine_step() && "ipm affine step computation not ended");
    for (auto slot : k_pair_slots) {
        positivity::apply_pair_step(d.elastic.value[slot], d.elastic.d_value[slot],
                                    ls.alpha_primal, d.elastic.dual[slot], d.elastic.d_dual[slot],
                                    pair_dual_alpha);
    }
    d.multiplier_.noalias() += multiplier_dual_alpha * d.d_multiplier_;
    if (d.ipm_cfg->ipm_accept_corrector()) {
        for (auto slot : k_pair_slots) {
            d.elastic.value[slot] = d.elastic.value[slot].array().max(1e-20);
            d.elastic.dual[slot] = d.elastic.dual[slot].array().max(1e-20);
        }
    }
}

void update_pair_ls_bounds_like_ipm(resto_eq_elastic_constr::approx_data &d,
                                    workspace_data *cfg) {
    auto &ls = cfg->as<linesearch_config>();
    for (auto slot : k_pair_slots) {
        positivity::update_pair_bounds(ls, d.elastic.value[slot], d.elastic.d_value[slot],
                                       d.elastic.dual[slot], d.elastic.d_dual[slot]);
    }
}

void backup_trial_pairs_like_ipm(resto_eq_elastic_constr::approx_data &d) {
    for (auto slot : k_pair_slots) {
        positivity::backup_pair(d.elastic.value[slot], d.elastic.value_backup[slot],
                                d.elastic.dual[slot], d.elastic.dual_backup[slot]);
    }
    d.multiplier_backup = d.multiplier_;
}

void restore_trial_pairs_like_ipm(resto_eq_elastic_constr::approx_data &d) {
    for (auto slot : k_pair_slots) {
        positivity::restore_pair(d.elastic.value[slot], d.elastic.value_backup[slot],
                                 d.elastic.dual[slot], d.elastic.dual_backup[slot]);
    }
    d.multiplier_ = d.multiplier_backup;
}

scalar_t objective_penalty_from_pairs(const resto_eq_elastic_constr::approx_data &d) {
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.value[slot].sum();
    }
    return rho_value(d, "objective_penalty_from_pairs") * sum;
}

scalar_t objective_penalty_dir_deriv_from_pairs(const resto_eq_elastic_constr::approx_data &d) {
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.d_value[slot].sum();
    }
    return rho_value(d, "objective_penalty_dir_deriv_from_pairs") * sum;
}

scalar_t search_penalty_from_pairs(const resto_eq_elastic_constr::approx_data &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.value[slot].array().max(scalar_t(1e-16)).log().sum();
    }
    return d.ipm_cfg->mu * sum;
}

scalar_t search_penalty_dir_deriv_from_pairs(const resto_eq_elastic_constr::approx_data &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += (d.elastic.d_value[slot].array() /
                d.elastic.value_backup[slot].array().max(scalar_t(1e-16)))
                   .sum();
    }
    return d.ipm_cfg->mu * sum;
}

} // namespace

resto_prox_cost::resto_prox_cost(const std::string &name,
                                 const var_list &u_args,
                                 const var_list &y_args,
                                 scalar_t rho_u,
                                 scalar_t rho_y)
    : generic_cost(name, approx_order::second),
      rho_u_(rho_u),
      rho_y_(rho_y) {
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

void resto_prox_cost::finalize_impl() {
    generic_cost::finalize_impl();
    for (size_t i = 0; i < in_args().size(); ++i) {
        for (size_t j = 0; j < in_args().size(); ++j) {
            hess_sp_[i][j].pattern = i == j ? sparsity::diag : sparsity::unknown;
        }
    }
}

void resto_prox_cost::value_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    scalar_t value = 0.;
    for (const sym &arg : in_args()) {
        auto &ref_field = arg.field() == __u ? d.u_ref : d.y_ref;
        auto &sigma_field = arg.field() == __u ? d.sigma_u_sq : d.sigma_y_sq;
        const auto x = d[arg];
        auto ref = d.problem()->extract(ref_field, arg);
        auto sigma = d.problem()->extract_tangent(sigma_field, arg);
        if (x.size() == 0 || ref.size() == 0 || sigma.size() == 0) {
            continue;
        }
        const vector delta = compute_tangent_delta(arg, x, ref);
        value += scalar_t(0.5) * sigma.dot(delta.cwiseProduct(delta)) * std::sqrt(d.mu[0]);
    }
    d.v_(0) += value;
}

void resto_prox_cost::jacobian_impl(func_approx_data &data) const {
    auto &d = data.as<approx_data>();
    size_t arg_idx = 0;
    for (const sym &arg : in_args()) {
        if (!d.has_jacobian_block(arg_idx)) {
            ++arg_idx;
            continue;
        }
        auto grad = d.jac(arg_idx);
        auto &ref_field = arg.field() == __u ? d.u_ref : d.y_ref;
        auto &sigma_field = arg.field() == __u ? d.sigma_u_sq : d.sigma_y_sq;
        auto ref = d.problem()->extract(ref_field, arg);
        auto sigma = d.problem()->extract_tangent(sigma_field, arg);
        const vector delta = compute_tangent_delta(arg, d[arg], ref);
        grad.array() += 0.5 * (sigma.array() * delta.array()).transpose() * std::sqrt(d.mu[0]);
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
        auto &sigma_field = arg.field() == __u ? d.sigma_u_sq : d.sigma_y_sq;
        auto sigma = d.problem()->extract_tangent(sigma_field, arg);
        block += sigma * std::sqrt(d.mu[0]);
    }
}

resto_eq_elastic_constr::resto_eq_elastic_constr(const std::string &name,
                                                 const constr &source)
    : soft_constr(name, approx_order::second, source->dim()),
      source_(source),
      source_func_(dynamic_cast<const generic_func *>(source.get())) {
    if (source_func_ == nullptr) {
        throw std::runtime_error(fmt::format("resto_eq_elastic_constr source {} is not a generic_func", source->name()));
    }
    field_hint_.is_eq = true;
    field_hint_.is_soft = true;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(source_func_->in_args());
    solver::overlay::copy_source_sparsity(*this, *source_func_);
}

void resto_eq_elastic_constr::setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {
    soft_constr::setup_workspace_data(data, ws_data);
    auto &d = data.as<approx_data>();
    d.ipm_cfg = &ws_data->as<solver::ipm_config>();
    d.rho = &ws_data->as<restoration_overlay_settings>().rho_eq;
}

func_approx_data_ptr_t resto_eq_elastic_constr::create_approx_data(sym_data &primal,
                                                                   lag_data &raw,
                                                                   shared_data &shared) const {
    std::unique_ptr<soft_constr::approx_data> base_d(make_approx<soft_constr>(primal, raw, shared));
    return std::make_unique<approx_data>(std::move(*base_d));
}

void resto_eq_elastic_constr::value_impl(func_approx_data &data) const {
    source_func_->value(data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    solver::ineq_soft::ensure_initialized(*this, d);
    if (local_state_dim(d.elastic) == 0) {
        // The restoration entry path does one value-only evaluation before the
        // soft-constraint initializer sizes and seeds the local elastic state.
        d.v_ = d.base_residual;
        return;
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_eq_elastic_constr::value_impl");
    d.v_ = d.base_residual;
    for (size_t k : range(k_pair_slots.size())) {
        d.v_.array() += k_pair_signs[k] * d.elastic.value[k_pair_slots[k]].array();
    }
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_eq_elastic_constr::value_impl requires ipm_cfg");
    }
    resto_eq_elastic_constr::compute_local_model(d.elastic, d.base_residual, d.multiplier_, rho_value(d, "resto_eq_elastic_constr::value_impl"), d.ipm_cfg->mu);
}

void resto_eq_elastic_constr::jacobian_impl(func_approx_data &data) const {
    source_func_->jacobian(data);
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_eq_elastic_constr::jacobian_impl requires ipm_cfg");
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_eq_elastic_constr::jacobian_impl");
    if (d.ipm_cfg->disable_corrections) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    resto_eq_elastic_constr::compute_local_model(d.elastic, d.base_residual, d.multiplier_, rho_value(d, "resto_eq_elastic_constr::jacobian_impl"), target_mu);
    propagate_jacobian(d, d.elastic.schur_rhs);
    propagate_hessian(d, d.elastic.schur_inv_diag);
}

void resto_eq_elastic_constr::hessian_impl(func_approx_data &data) const {
    if (source_func_->order() >= approx_order::second) {
        source_func_->hessian(data);
    }
}

void resto_eq_elastic_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_eq_elastic_constr::initialize requires ipm_cfg");
    }
    const scalar_t rho = rho_value(d, "resto_eq_elastic_constr::initialize");
    if (!(rho > 0.)) {
        throw std::runtime_error("resto_eq_elastic_constr::initialize requires rho > 0");
    }
    const scalar_t mu_bar = d.ipm_cfg->mu;
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("resto_eq_elastic_constr::initialize requires mu > 0");
    }
    resto_eq_elastic_constr::resize_local_state(d.elastic, 0, d.func_.dim());
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const scalar_t c = d.base_residual(i);
        const scalar_t disc = (mu_bar - rho * c) * (mu_bar - rho * c) + scalar_t(2.) * rho * mu_bar * c;
        const scalar_t sqrt_disc = std::sqrt(std::max(disc, scalar_t(0.)));
        const scalar_t n = (mu_bar - rho * c + sqrt_disc) / (scalar_t(2.) * rho);
        const scalar_t p = c + n;
        const scalar_t p_clamped = std::max(p, scalar_t(1e-16));
        const scalar_t n_clamped = std::max(n, scalar_t(1e-16));
        const scalar_t z_p = mu_bar / p_clamped;
        const scalar_t z_n = mu_bar / n_clamped;

        d.elastic.value[detail::slot_p](i) = p_clamped;
        d.elastic.value[detail::slot_n](i) = n_clamped;
        d.elastic.dual[detail::slot_p](i) = z_p;
        d.elastic.dual[detail::slot_n](i) = z_n;
    }
    d.multiplier_.setZero();
    d.multiplier_backup = d.multiplier_;
    resto_eq_elastic_constr::compute_local_model(d.elastic, d.base_residual, d.multiplier_, rho, mu_bar);
}

void resto_eq_elastic_constr::finalize_newton_step(data_map_t &data) const {
    finalize_pair_newton_step(data.as<approx_data>());
}

void resto_eq_elastic_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    finalize_predictor_pairs_like_ipm(d, cfg);
}

void resto_eq_elastic_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_eq_elastic_constr::apply_corrector_step requires ipm_cfg");
    }
    apply_corrector_pairs_like_ipm(d);
    const auto *corrector = d.ipm_cfg->ipm_accept_corrector() ? &d.elastic.corrector : nullptr;
    resto_eq_elastic_constr::compute_local_model(d.elastic,
                                                 d.base_residual,
                                                 d.multiplier_,
                                                 rho_value(d, "resto_eq_elastic_constr::apply_corrector_step"),
                                                 d.ipm_cfg->mu,
                                                 corrector);
    propagate_jacobian(d, d.elastic.schur_rhs);
}

void resto_eq_elastic_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &ls = cfg->as<linesearch_config>();
    apply_affine_pairs_like_ipm(d, cfg, ls.dual_alpha_for_eq(), ls.dual_alpha_for_ineq());
}

void resto_eq_elastic_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    update_pair_ls_bounds_like_ipm(d, cfg);
}

void resto_eq_elastic_constr::backup_trial_state(data_map_t &data) const {
    backup_trial_pairs_like_ipm(data.as<approx_data>());
}

void resto_eq_elastic_constr::restore_trial_state(data_map_t &data) const {
    restore_trial_pairs_like_ipm(data.as<approx_data>());
}

scalar_t resto_eq_elastic_constr::objective_penalty(const func_approx_data &data) const {
    return objective_penalty_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_eq_elastic_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    return objective_penalty_dir_deriv_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_eq_elastic_constr::search_penalty(const func_approx_data &data) const {
    return search_penalty_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_eq_elastic_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    return search_penalty_dir_deriv_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_eq_elastic_constr::local_stat_residual_inf(const func_approx_data &data) const {
    // Equality-elastic local stationarity:
    // max(||rho - lambda - z_p||_inf, ||rho + lambda - z_n||_inf).
    const auto &d = static_cast<const approx_data &>(data);
    require_local_state_initialized(d.elastic, data.func_.dim(), "resto_eq_elastic_constr::local_stat_residual_inf");
    return resto_eq_elastic_constr::current_local_residuals(d.elastic).inf_stat;
}

scalar_t resto_eq_elastic_constr::local_comp_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    require_local_state_initialized(d.elastic, data.func_.dim(), "resto_eq_elastic_constr::local_comp_residual_inf");
    return resto_eq_elastic_constr::current_local_residuals(d.elastic).inf_comp;
}

resto_ineq_elastic_ipm_constr::resto_ineq_elastic_ipm_constr(const std::string &name,
                                                             const constr &source)
    : ineq_constr(name, approx_order::second, source->dim()),
      source_(source),
      source_func_(dynamic_cast<const generic_func *>(source.get())) {
    if (source_func_ == nullptr) {
        throw std::runtime_error(fmt::format("resto_ineq_elastic_ipm_constr source {} is not a generic_func", source->name()));
    }
    field_hint_.is_eq = false;
    set_default_hess_sparsity(sparsity::dense);
    add_arguments(source_func_->in_args());
    solver::overlay::copy_source_sparsity(*this, *source_func_);
    if (const auto *src_ineq = dynamic_cast<const ineq_constr *>(source.get())) {
        if (const auto *box = src_ineq->box_info(); box != nullptr) {
            set_box_info(std::make_shared<box_spec>(*box));
        } else {
            synthesize_upper_half_box_info_if_missing();
        }
    } else {
        synthesize_upper_half_box_info_if_missing();
    }
}

void resto_ineq_elastic_ipm_constr::setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {
    ineq_constr::setup_workspace_data(data, ws_data);
    auto &d = data.as<approx_data>();
    d.ipm_cfg = &ws_data->as<solver::ipm_config>();
    d.rho = &ws_data->as<restoration_overlay_settings>().rho_ineq;
}

func_approx_data_ptr_t resto_ineq_elastic_ipm_constr::create_approx_data(sym_data &primal,
                                                                         lag_data &raw,
                                                                         shared_data &shared) const {
    std::unique_ptr<ineq_constr::approx_data> base_d(make_approx<ineq_constr>(primal, raw, shared));
    return std::make_unique<approx_data>(std::move(*base_d));
}

void resto_ineq_elastic_ipm_constr::value_impl(func_approx_data &data) const {
    source_func_->value(data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    solver::ineq_soft::ensure_initialized(*this, d);
    // Local restoration complementarity is summarized directly from the wrapper's
    // current local model. Keep lag_data::comp_ neutral so public stage comp
    // statistics are not polluted by restoration-only bookkeeping.
    d.comp_.setZero();
    if (local_state_dim(d.elastic) == 0) {
        // The restoration entry path does one value-only evaluation before the
        // soft-constraint initializer sizes and seeds the local elastic state.
        d.v_ = d.base_residual;
        return;
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::value_impl");
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::value_impl requires ipm_cfg");
    }
    refresh_ineq_local_model(d, d.require_box_spec("resto_ineq_elastic_ipm_constr::value_impl"),
                             "resto_ineq_elastic_ipm_constr::value_impl", d.ipm_cfg->mu);
}

void resto_ineq_elastic_ipm_constr::jacobian_impl(func_approx_data &data) const {
    source_func_->jacobian(data);
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::jacobian_impl requires ipm_cfg");
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::jacobian_impl");
    if (d.ipm_cfg->disable_corrections) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    refresh_ineq_local_model(d, d.require_box_spec("resto_ineq_elastic_ipm_constr::jacobian_impl"),
                             "resto_ineq_elastic_ipm_constr::jacobian_impl", target_mu);
    propagate_jacobian(d, d.elastic.schur_rhs_net);
    propagate_hessian(d, d.elastic.schur_inv_diag_sum);
}

void resto_ineq_elastic_ipm_constr::hessian_impl(func_approx_data &data) const {
    if (source_func_->order() >= approx_order::second) {
        source_func_->hessian(data);
    }
}

void resto_ineq_elastic_ipm_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::initialize requires ipm_cfg");
    }
    resto_ineq_elastic_ipm_constr::resize_local_state(d.elastic, d.func_.dim(), 0);
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::initialize");
    const scalar_t rho = rho_value(d, "resto_ineq_elastic_ipm_constr::initialize");
    const scalar_t mu_bar = d.ipm_cfg->mu;
    if (!(rho > 0.)) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::initialize requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::initialize requires mu > 0");
    }
    update_ineq_side_residuals(d);
    for_each_active_ineq_side(d.elastic, box, [&](auto side, auto &side_state, const auto &mask) {
        for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
            if (!mask(i)) {
                continue;
            }
            const scalar_t t0 = d.slack_init[side](i);
            const scalar_t nu_t0 = d.multiplier_init[side](i);
            if (!(t0 > 0.)) {
                throw std::runtime_error("resto_ineq_elastic_ipm_constr::initialize requires t0 > 0");
            }
            if (!(nu_t0 > 0.)) {
                throw std::runtime_error("resto_ineq_elastic_ipm_constr::initialize requires nu_t0 > 0");
            }
            const scalar_t c = side_state.residual(i) + t0;
            const scalar_t disc = (mu_bar - rho * c) * (mu_bar - rho * c) + scalar_t(2.) * rho * mu_bar * c;
            const scalar_t sqrt_disc = std::sqrt(std::max(disc, scalar_t(0.)));
            const scalar_t n = std::max((mu_bar - rho * c + sqrt_disc) / (scalar_t(2.) * rho), scalar_t(1e-16));
            const scalar_t p = std::max(c + n, scalar_t(1e-16));
            side_state.value[detail::slot_t](i) = t0;
            side_state.dual[detail::slot_t](i) = nu_t0;
            side_state.value[detail::slot_p](i) = p;
            side_state.value[detail::slot_n](i) = n;
            side_state.dual[detail::slot_p](i) = mu_bar / p;
            side_state.dual[detail::slot_n](i) = mu_bar / n;
        }
    });
    d.multiplier_backup = d.multiplier_;
    resto_ineq_elastic_ipm_constr::compute_local_model(d.elastic, box, rho, mu_bar);
    sync_ineq_overlay_views(d);
    d.multiplier_backup = d.multiplier_;
}

void resto_ineq_elastic_ipm_constr::finalize_newton_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::finalize_newton_step");
    d.d_multiplier_.setZero();
    size_t arg_idx = 0;
    for (const sym &arg : d.func_.in_args()) {
        if (arg.field() < field::num_prim) {
            d.jac_step.noalias() = d.jac_[arg_idx] * d.prim_step_[arg_idx];
            d.d_multiplier_.noalias() += d.jac_step;
        }
        ++arg_idx;
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::finalize_newton_step");
    for_each_active_ineq_side(d.elastic, box, [&](auto side, auto &side_state, const auto &mask) {
        const auto masked_delta =
            mask.select(side_jac_sign(side) * d.d_multiplier_.array(), scalar_t(0)).matrix();
        side_state.d_dual[detail::slot_t] =
            side_state.schur_inv_diag.array() * (masked_delta.array() + side_state.condensed_rhs.array());
        side_state.d_value[detail::slot_t] =
            -(side_state.value[detail::slot_t].array() / side_state.dual[detail::slot_t].array()) *
                side_state.d_dual[detail::slot_t].array() -
            side_state.backsub_rhs[detail::slot_t].array();
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            const scalar_t sign = k_pair_signs[k];
            side_state.d_value[slot] =
                -sign * (side_state.value[slot].array() / side_state.dual[slot].array()) *
                    side_state.d_dual[detail::slot_t].array() -
                side_state.backsub_rhs[slot].array();
            side_state.d_dual[slot] = side_state.r_stat[slot].array() + sign * side_state.d_dual[detail::slot_t].array();
        }
    });
    sync_ineq_overlay_views(d);
}

void resto_ineq_elastic_ipm_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &worker = cfg->as<solver::ipm_config::worker_type>();
    auto &ls = cfg->as<linesearch_config>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::finalize_predictor_step");
    assert(d.ipm_cfg != nullptr);
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    for_each_active_ineq_side_slot(d.elastic, box, [&](auto, auto slot, auto &side_state, const auto &mask) {
        worker.n_ipm_cstr += static_cast<size_t>(mask.count());
        worker.prev_aff_comp +=
            mask.select(side_state.dual[slot].array() * side_state.value[slot].array(), scalar_t(0)).sum();
        side_state.corrector[slot].array() =
            ls.alpha_dual * side_state.d_dual[slot].array() * ls.alpha_primal * side_state.d_value[slot].array();
        worker.post_aff_comp +=
            mask.select((side_state.dual[slot] + ls.alpha_dual * side_state.d_dual[slot]).array() *
                            (side_state.value[slot] + ls.alpha_primal * side_state.d_value[slot]).array(),
                        scalar_t(0))
                .sum();
    });
}

void resto_ineq_elastic_ipm_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::apply_corrector_step requires ipm_cfg");
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::apply_corrector_step");
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        for_each_active_ineq_side_slot(d.elastic, box, [](auto, auto slot, auto &side_state, const auto &) {
            side_state.corrector[slot].setZero();
        });
    }
    detail::elastic_side_array<detail::elastic_triplet_array<vector>> corrector;
    const auto *corrector_ptr = d.ipm_cfg->ipm_accept_corrector() ? &corrector : nullptr;
    if (corrector_ptr != nullptr) {
        for_each_active_ineq_side(d.elastic, box, [&](auto side, const auto &side_state, const auto &) {
            corrector[side] = side_state.corrector;
        });
    }
    refresh_ineq_local_model(d, box, "resto_ineq_elastic_ipm_constr::apply_corrector_step", d.ipm_cfg->mu, corrector_ptr);
    propagate_jacobian(d, d.elastic.schur_rhs_net);
}

void resto_ineq_elastic_ipm_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &ls = cfg->as<linesearch_config>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::apply_affine_step");
    for_each_active_ineq_side_slot(d.elastic, box, [&](auto, auto slot, auto &side_state, const auto &mask) {
        positivity::apply_pair_step(side_state.value[slot], side_state.d_value[slot], ls.alpha_primal,
                                    side_state.dual[slot], side_state.d_dual[slot], ls.dual_alpha_for_ineq());
        side_state.value[slot] =
            mask.select(side_state.value[slot].array().max(1e-20), scalar_t(0)).matrix();
        side_state.dual[slot] =
            mask.select(side_state.dual[slot].array().max(1e-20), scalar_t(0)).matrix();
    });
    refresh_ineq_local_model(d, box, "resto_ineq_elastic_ipm_constr::apply_affine_step",
                             d.ipm_cfg != nullptr ? d.ipm_cfg->mu : scalar_t(0));
}

void resto_ineq_elastic_ipm_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &ls = cfg->as<linesearch_config>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::update_ls_bounds");
    for_each_active_ineq_side_slot(d.elastic, box, [&](auto, auto slot, auto &side_state, const auto &) {
        positivity::update_pair_bounds(ls, side_state.value[slot], side_state.d_value[slot],
                                       side_state.dual[slot], side_state.d_dual[slot]);
    });
}

void resto_ineq_elastic_ipm_constr::backup_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::backup_trial_state");
    for_each_active_ineq_side_slot(d.elastic, box, [](auto, auto slot, auto &side_state, const auto &) {
        positivity::backup_pair(side_state.value[slot], side_state.value_backup[slot],
                                side_state.dual[slot], side_state.dual_backup[slot]);
    });
    d.multiplier_backup = d.multiplier_;
}

void resto_ineq_elastic_ipm_constr::restore_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::restore_trial_state");
    for_each_active_ineq_side_slot(d.elastic, box, [](auto, auto slot, auto &side_state, const auto &) {
        positivity::restore_pair(side_state.value[slot], side_state.value_backup[slot],
                                 side_state.dual[slot], side_state.dual_backup[slot]);
    });
    refresh_ineq_local_model(d, box, "resto_ineq_elastic_ipm_constr::restore_trial_state",
                             d.ipm_cfg != nullptr ? d.ipm_cfg->mu : scalar_t(0));
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    scalar_t sum = 0.;
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::objective_penalty");
    for_each_active_ineq_side(d.elastic, box, [&](auto, const auto &side_state, const auto &) {
        sum += side_state.value[detail::slot_p].sum() + side_state.value[detail::slot_n].sum();
    });
    return rho_value(d, "resto_ineq_elastic_ipm_constr::objective_penalty") * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    scalar_t sum = 0.;
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv");
    for_each_active_ineq_side(d.elastic, box, [&](auto, const auto &side_state, const auto &) {
        sum += side_state.d_value[detail::slot_p].sum() + side_state.d_value[detail::slot_n].sum();
    });
    return rho_value(d, "resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv") * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::search_penalty");
    scalar_t sum = 0.;
    for_each_active_ineq_side_slot(d.elastic, box, [&](auto, auto slot, const auto &side_state, const auto &mask) {
        const auto safe = mask.select(side_state.value[slot].array(), scalar_t(1));
        sum += mask.select(safe.log(), scalar_t(0)).sum();
    });
    return d.ipm_cfg->mu * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv");
    scalar_t sum = 0.;
    for_each_active_ineq_side_slot(d.elastic, box, [&](auto, auto slot, const auto &side_state, const auto &mask) {
        const auto safe = mask.select(side_state.value_backup[slot].array(), scalar_t(1));
        sum += mask.select(side_state.d_value[slot].array() / safe, scalar_t(0)).sum();
    });
    return d.ipm_cfg->mu * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::local_stat_residual_inf(const func_approx_data &data) const {
    // Inequality-elastic local stationarity:
    // max(||rho - nu_t - nu_p||_inf, ||rho + nu_t - nu_n||_inf).
    const auto &d = static_cast<const approx_data &>(data);
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::local_stat_residual_inf");
    return current_local_residuals(d.elastic).inf_stat;
}

scalar_t resto_ineq_elastic_ipm_constr::local_comp_residual_inf(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::local_comp_residual_inf");
    return current_local_residuals(d.elastic).inf_comp;
}

ocp_ptr_t build_restoration_overlay_problem(const ocp_ptr_t &source_prob,
                                            const restoration_overlay_settings &settings) {
    ocp::active_status_config config;
    for (auto field : std::array{__cost, __eq_x, __eq_xu, __ineq_x, __ineq_xu, __eq_x_soft, __eq_xu_soft}) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            config.deactivate_list.emplace_back(*expr);
        }
    }

    auto resto_prob = source_prob->clone(config);
    var_list u_args;
    var_list y_args;
    for (const sym &arg : resto_prob->exprs(__u)) {
        u_args.emplace_back(arg);
    }
    for (const sym &arg : resto_prob->exprs(__y)) {
        y_args.emplace_back(arg);
    }
    if (!u_args.empty() || !y_args.empty()) {
        auto prox = cost(new resto_prox_cost("resto_prox", u_args, y_args, settings.rho_u, settings.rho_y));
        resto_prob->add(*prox);
    }

    solver::overlay::add_constr_overlay_group(source_prob, resto_prob, std::array{__eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}, [&](const constr &source) {
        return constr(new resto_eq_elastic_constr(solver::overlay::overlay_name(*source, "resto_eq"),
                                                  source));
    });

    if (settings.rho_ineq > scalar_t(0.)) {
        solver::overlay::add_constr_overlay_group(source_prob, resto_prob, std::array{__ineq_x, __ineq_xu},
                                                  [&](const constr &source) {
                                                      return constr(new resto_ineq_elastic_ipm_constr(
                                                          solver::overlay::overlay_name(*source, "resto_ineq"),
                                                          source));
                                                  });
    }

    resto_prob->wait_until_ready();
    return resto_prob;
}

void sync_outer_to_restoration_state(node_data &outer,
                                     node_data &resto,
                                     scalar_t prox_eps,
                                     scalar_t *mu) {
    solver::overlay::copy_primal_and_params(outer, resto);

    solver::overlay::copy_dense_duals_if_present(outer, resto, hard_constr_fields);
    solver::overlay::copy_source_multipliers<resto_eq_elastic_constr>(outer, resto, std::array{__eq_x_soft, __eq_xu_soft});
    solver::overlay::for_each_overlay_field<resto_ineq_elastic_ipm_constr>(
        resto, std::array{__ineq_x, __ineq_xu}, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
            solver::overlay::copy_source_multiplier(d.multiplier_, outer, overlay);
            copy_ineq_side_init(d.slack_init, d.multiplier_init, outer, overlay);
        });

    resto.for_each(__cost, [&](const resto_prox_cost &c, resto_prox_cost::approx_data &d) {
        d.u_ref = d.primal_->value_[__u];
        d.y_ref = d.primal_->value_[__y];
        d.sigma_u_sq.resize(d.problem()->tdim(__u));
        d.sigma_u_sq.setZero();
        for (const sym &arg : d.problem()->exprs(__u)) {
            fill_sigma_tangent(arg,
                               d.problem()->extract(d.u_ref, arg),
                               d.problem()->extract_tangent(d.sigma_u_sq, arg),
                               prox_eps,
                               c.rho_u());
        }
        d.sigma_y_sq.resize(d.problem()->tdim(__y));
        d.sigma_y_sq.setZero();
        for (const sym &arg : d.problem()->exprs(__y)) {
            fill_sigma_tangent(arg,
                               d.problem()->extract(d.y_ref, arg),
                               d.problem()->extract_tangent(d.sigma_y_sq, arg),
                               prox_eps,
                               c.rho_y());
        }
        d.mu = mu;
    });
}

namespace {

void copy_restoration_primal_to_outer(node_data &resto,
                                      node_data &outer) {
    solver::overlay::copy_primal_and_params(resto, outer);
}

struct outer_boxed_ipm_view {
    solver::ipm_constr::ipm_data &data;
    const ineq_constr::box_spec &box;
};

outer_boxed_ipm_view outer_boxed_ipm(node_data &outer,
                                     const resto_ineq_elastic_ipm_constr &overlay,
                                     std::string_view where) {
    auto &outer_ipm = outer.data(overlay.source()).as<solver::ipm_constr::ipm_data>();
    const auto *outer_constr = dynamic_cast<const ineq_constr *>(&outer_ipm.func_);
    const auto *box = outer_constr != nullptr ? outer_constr->box_info() : nullptr;
    if (box == nullptr) {
        throw std::runtime_error(fmt::format("boxed ipm missing box_info in {}", where));
    }
    return {outer_ipm, *box};
}

template <typename BeginFn, typename SideFn>
void for_each_restoration_ineq_side(node_data &resto,
                                    node_data &outer,
                                    std::string_view where,
                                    BeginFn &&begin_fn,
                                    SideFn &&side_fn) {
    solver::overlay::for_each_overlay_field<resto_ineq_elastic_ipm_constr>(
        resto, std::array{__ineq_x, __ineq_xu}, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &overlay_data) {
            auto outer_view = outer_boxed_ipm(outer, overlay, where);
            begin_fn(outer_view.data, overlay_data);
            for_each_active_ineq_side(overlay_data.elastic, outer_view.box, [&](auto side, const auto &resto_side, const auto &mask) {
                side_fn(outer_view.data, *outer_view.data.box_side_[side], side, resto_side, mask, overlay_data);
            });
        });
}

void copy_restoration_candidate_slack_to_outer(node_data &resto,
                                               node_data &outer) {
    for_each_restoration_ineq_side(resto, outer, "restoration candidate sync",
        [](auto &, const auto &) {},
        [](auto &, auto &outer_pair, auto, const auto &resto_side, const auto &mask, const auto &) {
            outer_pair.slack =
                mask.select(resto_side.value[detail::slot_t].array(), scalar_t(0)).matrix();
            outer_pair.slack_backup = outer_pair.slack;
            outer_pair.multiplier_backup = outer_pair.multiplier;
            outer_pair.d_slack.setZero();
            outer_pair.d_multiplier.setZero();
        });
}

void copy_restoration_equality_duals_to_outer(node_data &resto,
                                              node_data &outer) {
    solver::overlay::copy_dense_duals_if_present(resto, outer, hard_constr_fields);
    solver::overlay::commit_source_multipliers<resto_eq_elastic_constr>(outer, resto, std::array{__eq_x_soft, __eq_xu_soft});
}

void copy_restoration_ineq_commit_to_outer(node_data &resto,
                                           node_data &outer) {
    for_each_restoration_ineq_side(resto, outer, "restoration sync",
        [](auto &outer_ipm, const auto &) {
            outer_ipm.multiplier_.setZero();
            outer_ipm.d_multiplier_.setZero();
            outer_ipm.comp_.setZero();
            outer_ipm.v_.setConstant(-std::numeric_limits<scalar_t>::infinity());
        },
        [](auto &outer_ipm, auto &outer_pair, auto side, const auto &resto_side, const auto &mask, const auto &overlay_data) {
            const auto target_slack = resto_side.value[detail::slot_t].array();
            const scalar_t mu = overlay_data.ipm_cfg->mu;
            outer_pair.d_slack =
                mask.select(target_slack - outer_pair.slack.array(), scalar_t(0)).matrix();
            outer_pair.d_multiplier =
                mask.select((mu - outer_pair.multiplier.array() * outer_pair.d_slack.array()) /
                                target_slack -
                                outer_pair.multiplier.array(),
                            scalar_t(0))
                    .matrix();
            outer_pair.slack = mask.select(target_slack, scalar_t(0)).matrix();
            outer_pair.slack_backup = outer_pair.slack;
            outer_pair.multiplier_backup = outer_pair.multiplier;
            static_cast<solver::ipm_constr::approx_data::side_data &>(outer_pair).r_s =
                mask.select(outer_pair.multiplier.array() * outer_pair.slack.array(), scalar_t(0)).matrix();
            const scalar_t dual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
            outer_ipm.multiplier_.array() +=
                dual_sign * mask.select(outer_pair.multiplier.array(), scalar_t(0));
            outer_ipm.d_multiplier_.array() +=
                dual_sign * mask.select(outer_pair.d_multiplier.array(), scalar_t(0));
            outer_ipm.v_ =
                mask.select(outer_ipm.v_.cwiseMax(resto_side.r_d).array(), outer_ipm.v_.array())
                    .matrix();
            outer_ipm.comp_ =
                mask.select((outer_pair.multiplier.array() * outer_pair.slack.array()).abs()
                                .max(outer_ipm.comp_.array()),
                            outer_ipm.comp_.array())
                    .matrix();
        });
}

} // namespace

void sync_restoration_candidate_to_outer_state(node_data &resto,
                                               node_data &outer) {
    copy_restoration_primal_to_outer(resto, outer);
    copy_restoration_candidate_slack_to_outer(resto, outer);
}

void commit_restoration_to_outer_state(node_data &resto, node_data &outer) {
    copy_restoration_primal_to_outer(resto, outer);
    copy_restoration_equality_duals_to_outer(resto, outer);
    copy_restoration_ineq_commit_to_outer(resto, outer);
}

} // namespace moto::solver::restoration
