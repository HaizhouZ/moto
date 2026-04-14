#include <moto/solver/restoration/resto_overlay.hpp>

#include <algorithm>
#include <cmath>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <string_view>
#include <tuple>

namespace moto::solver::restoration {

namespace {
constexpr std::array k_pair_slots{detail::slot_p, detail::slot_n};
constexpr std::array<scalar_t, 2> k_pair_signs{scalar_t(-1.), scalar_t(1.)};
constexpr std::array k_triplet_slots{detail::slot_t, detail::slot_p, detail::slot_n};

template <typename ApproxData>
constexpr auto active_slots() {
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        return k_pair_slots;
    } else {
        return k_triplet_slots;
    }
}

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

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

template <typename ApproxData>
scalar_t rho_value(const ApproxData &d, std::string_view where) {
    if (d.rho == nullptr) {
        throw std::runtime_error(fmt::format("{} requires restoration rho in workspace data", where));
    }
    return *d.rho;
}
} // namespace

namespace {

std::string overlay_name(const generic_func &source, std::string_view suffix) {
    return fmt::format("{}__{}", source.name(), suffix);
}

template <typename Fn>
void for_each_source_constr(const ocp_ptr_t &prob, field_t field, Fn &&fn) {
    for (const shared_expr &expr : prob->exprs(field)) {
        if (auto source = std::dynamic_pointer_cast<generic_constr>(expr)) {
            fn(constr(std::move(source)));
        }
    }
}

void copy_source_sparsity(generic_func &dst, const generic_func &src) {
    const auto &src_args = src.in_args();
    const auto &src_sp = src.jac_sparsity();
    for (size_t i = 0; i < src_args.size() && i < src_sp.size(); ++i) {
        dst.set_jac_sparsity(src_args[i], src_sp[i]);
    }
    dst.set_hess_sparsity(src.hess_sparsity());
}

template <typename Factory, size_t N>
void add_overlay_group(const ocp_ptr_t &source_prob,
                       ocp_ptr_t &resto_prob,
                       const std::array<field_t, N> &fields,
                       Factory &&factory) {
    for (auto field : fields) {
        for_each_source_constr(source_prob, field, [&](const constr &source) {
            resto_prob->add(*factory(source));
        });
    }
}

template <typename Overlay, typename Fn>
void for_each_overlay_match(node_data &node, field_t field, Fn &&fn) {
    node.for_each(field, [&](const Overlay &overlay, typename Overlay::approx_data &d) {
        fn(overlay, d);
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
    // if (arg.tdim() == arg.dim()) {
    sigma_sq = weight * ref.array().abs().max(eps).inverse().square().min(1.);
    return;
    // }
    // const scalar_t scale_ref = ref.size() > 0 ? ref.cwiseAbs().maxCoeff() : scalar_t(0.);
    // const scalar_t sigma = std::min(weight / std::pow(std::max(scale_ref, eps), 2), scalar_t(1.));
    // sigma_sq.setConstant(sigma);
}

vector compute_tangent_delta(const sym &arg, vector_ref x, vector_ref ref) {
    vector delta(arg.tdim());
    arg.difference(x, ref, delta);
    return delta;
}

template <typename Overlay>
void copy_dual_slice(vector_ref dst, const node_data &outer, const Overlay &overlay) {
    const auto &source_data = outer.data(overlay.source());
    dst = const_cast<func_approx_data &>(source_data).template as<generic_constr::approx_data>().multiplier_;
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
    for (auto side : box_sides) {
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
    for (auto side : box_sides) {
        if (box.bound_source[side] == ineq_constr::box_bound_source::constant) {
            bound_eval[side] = d.box_const_[side];
        } else if (box.bound_source[side] == ineq_constr::box_bound_source::in_arg) {
            bound_eval[side] = d[static_cast<const sym &>(box.bound_var[side])];
        } else {
            bound_eval[side].setZero(d.base_residual.size());
        }
        d.elastic.side[side].residual =
            box.present_mask[side]
                .select(side_jac_sign(side) * (d.base_residual.array() - bound_eval[side].array()), scalar_t(0))
                .matrix();
    }
}

void sync_ineq_overlay_views(resto_ineq_elastic_ipm_constr::approx_data &d) {
    const auto &box = d.require_box_spec("sync_ineq_overlay_views");
    d.v_ = d.elastic.primal_view;
    d.multiplier_.setZero();
    d.d_multiplier_.setZero();
    d.comp_.setZero();
    for (auto side : box_sides) {
        const auto &side_state = d.elastic.side[side];
        const scalar_t dual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
        d.multiplier_.array() +=
            dual_sign * box.present_mask[side].select(side_state.dual[detail::slot_t].array(), scalar_t(0));
        d.d_multiplier_.array() +=
            dual_sign * box.present_mask[side].select(side_state.d_dual[detail::slot_t].array(), scalar_t(0));
    }
}

template <typename ApproxData>
void finalize_predictor_pairs_like_ipm(ApproxData &d,
                                       workspace_data *cfg) {
    auto &worker = cfg->as<solver::ipm_config::worker_type>();
    auto &ls = cfg->as<linesearch_config>();
    assert(d.ipm_cfg != nullptr);
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    const scalar_t alpha_primal = ls.alpha_primal;
    const scalar_t alpha_dual = ls.alpha_dual;
    for (auto slot : active_slots<ApproxData>()) {
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
    for (auto slot : active_slots<ApproxData>()) {
        d.elastic.corrector[slot].array() =
            ls.alpha_dual * d.elastic.d_dual[slot].array() * ls.alpha_primal *
            d.elastic.d_value[slot].array();
    }
}

template <typename ApproxData>
void apply_corrector_pairs_like_ipm(ApproxData &d) {
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("apply_corrector_pairs_like_ipm requires ipm_cfg");
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "apply_corrector_pairs_like_ipm");
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        for (auto slot : active_slots<ApproxData>()) {
            d.elastic.corrector[slot].setZero();
        }
    }
}

template <typename ApproxData>
void finalize_pair_newton_step(ApproxData &d) {
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
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        const scalar_t eps = 1e-16;
        d.elastic.d_multiplier = d.elastic.schur_inv_diag.array() * (d.jac_step.array() + d.elastic.condensed_rhs.array());
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            const scalar_t sign = k_pair_signs[k];
            const auto scale = d.elastic.value[slot].array() / d.elastic.dual[slot].array().max(eps);
            d.elastic.d_value[slot] = -sign * scale * d.elastic.d_multiplier.array() - d.elastic.backsub_rhs[slot].array();
            d.elastic.d_dual[slot] = d.elastic.r_stat[slot].array() + sign * d.elastic.d_multiplier.array();
        }
    } else {
        d.elastic.d_dual[detail::slot_t] = d.elastic.schur_inv_diag.array() * (d.jac_step.array() + d.elastic.condensed_rhs.array());
        d.elastic.d_value[detail::slot_t] =
            -(d.elastic.value[detail::slot_t].array() / d.elastic.dual[detail::slot_t].array()) *
                d.elastic.d_dual[detail::slot_t].array() -
            d.elastic.backsub_rhs[detail::slot_t].array();
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            const scalar_t sign = k_pair_signs[k];
            d.elastic.d_value[slot] =
                -sign * (d.elastic.value[slot].array() / d.elastic.dual[slot].array()) * d.elastic.d_dual[detail::slot_t].array() -
                d.elastic.backsub_rhs[slot].array();
            d.elastic.d_dual[slot] = d.elastic.r_stat[slot].array() + sign * d.elastic.d_dual[detail::slot_t].array();
        }
    }
    if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
        d.d_multiplier_ = d.elastic.d_dual[detail::slot_t];
    } else if constexpr (requires { d.elastic.d_multiplier; }) {
        d.d_multiplier_ = d.elastic.d_multiplier;
    } else {
        d.d_multiplier_.setZero();
    }
}

template <typename ApproxData, typename InitFn>
void initialize_pair_overlay(ApproxData &d, InitFn &&init_fn) {
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const auto init = init_fn(i);
        d.elastic.value[detail::slot_p](i) = init.p;
        d.elastic.value[detail::slot_n](i) = init.n;
        if constexpr (requires { init.nu_p; }) {
            d.elastic.dual[detail::slot_p](i) = init.nu_p;
            d.elastic.dual[detail::slot_n](i) = init.nu_n;
        } else {
            d.elastic.dual[detail::slot_p](i) = init.z_p;
            d.elastic.dual[detail::slot_n](i) = init.z_n;
        }
        if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
            d.elastic.value[detail::slot_t](i) = init.t;
            d.elastic.dual[detail::slot_t](i) = init.nu_t;
            d.multiplier_(i) = init.nu_t;
        } else {
            d.multiplier_(i) = init.lambda;
        }
    }
    d.multiplier_backup = d.multiplier_;
}

template <typename ApproxData>
void apply_affine_pairs_like_ipm(ApproxData &d,
                                 workspace_data *cfg,
                                 scalar_t multiplier_dual_alpha,
                                 scalar_t pair_dual_alpha) {
    auto &ls = cfg->as<linesearch_config>();
    assert(d.ipm_cfg != nullptr);
    assert(!d.ipm_cfg->ipm_computing_affine_step() && "ipm affine step computation not ended");
    for (auto slot : active_slots<ApproxData>()) {
        positivity::apply_pair_step(d.elastic.value[slot], d.elastic.d_value[slot],
                                    ls.alpha_primal, d.elastic.dual[slot], d.elastic.d_dual[slot],
                                    pair_dual_alpha);
    }
    if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
        d.multiplier_ = d.elastic.dual[detail::slot_t];
    } else {
        d.multiplier_.noalias() += multiplier_dual_alpha * d.d_multiplier_;
    }
    if (d.ipm_cfg->ipm_accept_corrector()) {
        for (auto slot : active_slots<ApproxData>()) {
            d.elastic.value[slot] = d.elastic.value[slot].array().max(1e-20);
            d.elastic.dual[slot] = d.elastic.dual[slot].array().max(1e-20);
        }
        if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
            d.multiplier_ = d.elastic.dual[detail::slot_t];
        }
    }
}

template <typename ApproxData>
void update_pair_ls_bounds_like_ipm(ApproxData &d,
                                    workspace_data *cfg,
                                    std::string_view kind,
                                    std::string_view name) {
    auto &ls = cfg->as<linesearch_config>();
    for (auto slot : active_slots<ApproxData>()) {
        positivity::update_pair_bounds(ls, d.elastic.value[slot], d.elastic.d_value[slot],
                                       d.elastic.dual[slot], d.elastic.d_dual[slot]);
    }
}

template <typename ApproxData>
void backup_trial_pairs_like_ipm(ApproxData &d) {
    for (auto slot : active_slots<ApproxData>()) {
        positivity::backup_pair(d.elastic.value[slot], d.elastic.value_backup[slot],
                                d.elastic.dual[slot], d.elastic.dual_backup[slot]);
    }
    d.multiplier_backup = d.multiplier_;
}

template <typename ApproxData>
void restore_trial_pairs_like_ipm(ApproxData &d) {
    for (auto slot : active_slots<ApproxData>()) {
        positivity::restore_pair(d.elastic.value[slot], d.elastic.value_backup[slot],
                                 d.elastic.dual[slot], d.elastic.dual_backup[slot]);
    }
    d.multiplier_ = d.multiplier_backup;
}

template <typename ApproxData>
scalar_t objective_penalty_from_pairs(const ApproxData &d) {
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.value[slot].sum();
    }
    return rho_value(d, "objective_penalty_from_pairs") * sum;
}

template <typename ApproxData>
scalar_t objective_penalty_dir_deriv_from_pairs(const ApproxData &d) {
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.d_value[slot].sum();
    }
    return rho_value(d, "objective_penalty_dir_deriv_from_pairs") * sum;
}

template <typename ApproxData>
scalar_t search_penalty_from_pairs(const ApproxData &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    scalar_t sum = 0.;
    for (auto slot : active_slots<ApproxData>()) {
        sum += d.elastic.value[slot].array().max(scalar_t(1e-16)).log().sum();
    }
    return d.ipm_cfg->mu * sum;
}

template <typename ApproxData>
scalar_t search_penalty_dir_deriv_from_pairs(const ApproxData &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    scalar_t sum = 0.;
    for (auto slot : active_slots<ApproxData>()) {
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
        hess_sp_[i][i].pattern = sparsity::diag;
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
            source_(source) {
    field_hint().is_eq = true;
    field_hint().is_soft = true;
    set_default_hess_sparsity(sparsity::dense);
    const auto &src_func = dynamic_cast<const generic_func &>(*source);
    add_arguments(src_func.in_args());
    copy_source_sparsity(*this, src_func);
}

void resto_eq_elastic_constr::finalize_impl() {
    soft_constr::finalize_impl();
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
    dynamic_cast<const generic_func &>(*source_).value(data);
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
    dynamic_cast<const generic_func &>(*source_).jacobian(data);
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
    if (source_->order() >= approx_order::second) {
        dynamic_cast<const generic_func &>(*source_).hessian(data);
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
        const scalar_t lambda = rho - z_p;

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
    update_pair_ls_bounds_like_ipm(d, cfg, "eq", name());
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
            source_(source) {
    field_hint().is_eq = false;
    set_default_hess_sparsity(sparsity::dense);
    const auto &src_func = dynamic_cast<const generic_func &>(*source);
    add_arguments(src_func.in_args());
    copy_source_sparsity(*this, src_func);
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

local_residual_summary resto_ineq_elastic_ipm_constr::current_local_residuals(const approx_data &data) const {
    require_local_state_initialized(data.elastic, data.func_.dim(), "resto_ineq_elastic_ipm_constr::current_local_residuals");
    return resto_ineq_elastic_ipm_constr::current_local_residuals(data.elastic);
}

void resto_ineq_elastic_ipm_constr::value_impl(func_approx_data &data) const {
    dynamic_cast<const generic_func &>(*source_).value(data);
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
    update_ineq_side_residuals(d);
    resto_ineq_elastic_ipm_constr::compute_local_model(
        d.elastic,
        d.require_box_spec("resto_ineq_elastic_ipm_constr::value_impl"),
        rho_value(d, "resto_ineq_elastic_ipm_constr::value_impl"),
        d.ipm_cfg->mu);
    sync_ineq_overlay_views(d);
}

void resto_ineq_elastic_ipm_constr::jacobian_impl(func_approx_data &data) const {
    dynamic_cast<const generic_func &>(*source_).jacobian(data);
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::jacobian_impl requires ipm_cfg");
    }
    require_local_state_initialized(d.elastic, d.func_.dim(), "resto_ineq_elastic_ipm_constr::jacobian_impl");
    if (d.ipm_cfg->disable_corrections) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    update_ineq_side_residuals(d);
    resto_ineq_elastic_ipm_constr::compute_local_model(
        d.elastic,
        d.require_box_spec("resto_ineq_elastic_ipm_constr::jacobian_impl"),
        rho_value(d, "resto_ineq_elastic_ipm_constr::jacobian_impl"),
        target_mu);
    sync_ineq_overlay_views(d);
    propagate_jacobian(d, d.elastic.schur_rhs_net);
    propagate_hessian(d, d.elastic.schur_inv_diag_sum);
}

void resto_ineq_elastic_ipm_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        dynamic_cast<const generic_func &>(*source_).hessian(data);
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
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
            if (!box.present_mask[side](i)) {
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
    }
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
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        const auto masked_delta =
            box.present_mask[side].select(side_jac_sign(side) * d.d_multiplier_.array(), scalar_t(0)).matrix();
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
    }
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
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (auto slot : k_triplet_slots) {
            worker.n_ipm_cstr += static_cast<size_t>(box.present_mask[side].count());
            worker.prev_aff_comp +=
                box.present_mask[side].select(side_state.dual[slot].array() * side_state.value[slot].array(), scalar_t(0)).sum();
            side_state.corrector[slot].array() =
                ls.alpha_dual * side_state.d_dual[slot].array() * ls.alpha_primal * side_state.d_value[slot].array();
            worker.post_aff_comp +=
                box.present_mask[side]
                    .select((side_state.dual[slot] + ls.alpha_dual * side_state.d_dual[slot]).array() *
                                (side_state.value[slot] + ls.alpha_primal * side_state.d_value[slot]).array(),
                            scalar_t(0))
                    .sum();
        }
    }
}

void resto_ineq_elastic_ipm_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    if (d.ipm_cfg == nullptr) {
        throw std::runtime_error("resto_ineq_elastic_ipm_constr::apply_corrector_step requires ipm_cfg");
    }
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        for (auto side : box_sides) {
            auto &side_state = d.elastic.side[side];
            for (auto slot : k_triplet_slots) {
                side_state.corrector[slot].setZero();
            }
        }
    }
    update_ineq_side_residuals(d);
    detail::elastic_side_array<detail::elastic_triplet_array<vector>> corrector;
    const auto *corrector_ptr = d.ipm_cfg->ipm_accept_corrector() ? &corrector : nullptr;
    if (corrector_ptr != nullptr) {
        for (auto side : box_sides) {
            corrector[side] = d.elastic.side[side].corrector;
        }
    }
    resto_ineq_elastic_ipm_constr::compute_local_model(
        d.elastic,
        d.require_box_spec("resto_ineq_elastic_ipm_constr::apply_corrector_step"),
        rho_value(d, "resto_ineq_elastic_ipm_constr::apply_corrector_step"),
        d.ipm_cfg->mu,
        corrector_ptr);
    sync_ineq_overlay_views(d);
    propagate_jacobian(d, d.elastic.schur_rhs_net);
}

void resto_ineq_elastic_ipm_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &ls = cfg->as<linesearch_config>();
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::apply_affine_step");
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (auto slot : k_triplet_slots) {
            positivity::apply_pair_step(side_state.value[slot], side_state.d_value[slot], ls.alpha_primal,
                                        side_state.dual[slot], side_state.d_dual[slot], ls.dual_alpha_for_ineq());
            side_state.value[slot] =
                box.present_mask[side].select(side_state.value[slot].array().max(1e-20), scalar_t(0)).matrix();
            side_state.dual[slot] =
                box.present_mask[side].select(side_state.dual[slot].array().max(1e-20), scalar_t(0)).matrix();
        }
    }
    update_ineq_side_residuals(d);
    resto_ineq_elastic_ipm_constr::compute_local_model(
        d.elastic,
        box,
        rho_value(d, "resto_ineq_elastic_ipm_constr::apply_affine_step"),
        d.ipm_cfg != nullptr ? d.ipm_cfg->mu : scalar_t(0));
    sync_ineq_overlay_views(d);
}

void resto_ineq_elastic_ipm_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    auto &ls = cfg->as<linesearch_config>();
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (auto slot : k_triplet_slots) {
            positivity::update_pair_bounds(ls, side_state.value[slot], side_state.d_value[slot],
                                           side_state.dual[slot], side_state.d_dual[slot]);
        }
    }
}

void resto_ineq_elastic_ipm_constr::backup_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (auto slot : k_triplet_slots) {
            positivity::backup_pair(side_state.value[slot], side_state.value_backup[slot],
                                    side_state.dual[slot], side_state.dual_backup[slot]);
        }
    }
    d.multiplier_backup = d.multiplier_;
}

void resto_ineq_elastic_ipm_constr::restore_trial_state(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    for (auto side : box_sides) {
        auto &side_state = d.elastic.side[side];
        for (auto slot : k_triplet_slots) {
            positivity::restore_pair(side_state.value[slot], side_state.value_backup[slot],
                                     side_state.dual[slot], side_state.dual_backup[slot]);
        }
    }
    update_ineq_side_residuals(d);
    resto_ineq_elastic_ipm_constr::compute_local_model(
        d.elastic,
        d.require_box_spec("resto_ineq_elastic_ipm_constr::restore_trial_state"),
        rho_value(d, "resto_ineq_elastic_ipm_constr::restore_trial_state"),
        d.ipm_cfg != nullptr ? d.ipm_cfg->mu : scalar_t(0));
    sync_ineq_overlay_views(d);
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    scalar_t sum = 0.;
    for (auto side : box_sides) {
        sum += d.elastic.side[side].value[detail::slot_p].sum();
        sum += d.elastic.side[side].value[detail::slot_n].sum();
    }
    return rho_value(d, "resto_ineq_elastic_ipm_constr::objective_penalty") * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    scalar_t sum = 0.;
    for (auto side : box_sides) {
        sum += d.elastic.side[side].d_value[detail::slot_p].sum();
        sum += d.elastic.side[side].d_value[detail::slot_n].sum();
    }
    return rho_value(d, "resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv") * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::search_penalty");
    scalar_t sum = 0.;
    for (auto side : box_sides) {
        for (auto slot : k_triplet_slots) {
            const auto safe = box.present_mask[side].select(d.elastic.side[side].value[slot].array(), scalar_t(1));
            sum += box.present_mask[side].select(safe.log(), scalar_t(0)).sum();
        }
    }
    return d.ipm_cfg->mu * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    const auto &d = static_cast<const approx_data &>(data);
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    const auto &box = d.require_box_spec("resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv");
    scalar_t sum = 0.;
    for (auto side : box_sides) {
        for (auto slot : k_triplet_slots) {
            const auto safe =
                box.present_mask[side].select(d.elastic.side[side].value_backup[slot].array(), scalar_t(1));
            sum += box.present_mask[side].select(d.elastic.side[side].d_value[slot].array() / safe, scalar_t(0)).sum();
        }
    }
    return d.ipm_cfg->mu * sum;
}

scalar_t resto_ineq_elastic_ipm_constr::local_stat_residual_inf(const func_approx_data &data) const {
    // Inequality-elastic local stationarity:
    // max(||rho - nu_t - nu_p||_inf, ||rho + nu_t - nu_n||_inf).
    return current_local_residuals(static_cast<const approx_data &>(data)).inf_stat;
}

scalar_t resto_ineq_elastic_ipm_constr::local_comp_residual_inf(const func_approx_data &data) const {
    return current_local_residuals(static_cast<const approx_data &>(data)).inf_comp;
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
        auto prox = cost(new resto_prox_cost("resto_prox", u_args, y_args, settings.rho_u, settings.rho_y));
        resto_prob->add(*prox);
    }

    add_overlay_group(source_prob, resto_prob, std::array{__eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}, [&](const constr &source) {
        return constr(new resto_eq_elastic_constr(overlay_name(*source, "resto_eq"),
                                                  source));
    });

    if (settings.rho_ineq > scalar_t(0.)) {
        add_overlay_group(source_prob, resto_prob, std::array{__ineq_x, __ineq_xu},
                          [&](const constr &source) {
                              return constr(new resto_ineq_elastic_ipm_constr(
                                  overlay_name(*source, "resto_ineq"),
                                  source));
                          });
    }

    resto_prob->wait_until_ready();
    return resto_prob;
}

namespace {
} // namespace

void sync_outer_to_restoration_state(node_data &outer,
                                     node_data &resto,
                                     scalar_t prox_eps,
                                     scalar_t *mu) {
    for (auto field : primal_fields) {
        resto.sym_val().value_[field] = outer.sym_val().value_[field];
    }
    resto.sym_val().value_[__p] = outer.sym_val().value_[__p];

    for (auto field : hard_constr_fields) {
        if (resto.dense().dual_[field].size() == 0 || outer.dense().dual_[field].size() == 0) {
            continue;
        }
        resto.dense().dual_[field] = outer.dense().dual_[field];
    }
    for (auto field : std::array{__eq_x_soft, __eq_xu_soft}) {
        for_each_overlay_match<resto_eq_elastic_constr>(resto, field, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
            copy_dual_slice(d.multiplier_, outer, overlay);
        });
    }
    for (auto field : std::array{__ineq_x, __ineq_xu}) {
        for_each_overlay_match<resto_ineq_elastic_ipm_constr>(
            resto, field, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
                copy_dual_slice(d.multiplier_, outer, overlay);
                copy_ineq_side_init(d.slack_init, d.multiplier_init, outer, overlay);
            });
    }

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

void sync_restoration_to_outer_state(node_data &resto,
                                     node_data &outer) {
    for (auto field : primal_fields) {
        outer.sym_val().value_[field] = resto.sym_val().value_[field];
    }
    outer.sym_val().value_[__p] = resto.sym_val().value_[__p];

    for (auto field : hard_constr_fields) {
        if (resto.dense().dual_[field].size() == 0 || outer.dense().dual_[field].size() == 0) {
            continue;
        }
        outer.dense().dual_[field] = resto.dense().dual_[field];
    }

    for (auto field : std::array{__eq_x_soft, __eq_xu_soft}) {
        for_each_overlay_match<resto_eq_elastic_constr>(
            resto, field,
            [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
                outer.data(overlay.source()).as<generic_constr::approx_data>().multiplier_ = d.multiplier_;
            });
    }

    for (auto field : std::array{__ineq_x, __ineq_xu}) {
        for_each_overlay_match<resto_ineq_elastic_ipm_constr>(
            resto, field, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &overlay_data) {
                auto &outer_ipm = outer.data(overlay.source()).as<solver::ipm_constr::ipm_data>();
                const scalar_t mu = overlay_data.ipm_cfg->mu;
                const auto *constr = dynamic_cast<const ineq_constr *>(&outer_ipm.func_);
                const auto *box = constr != nullptr ? constr->box_info() : nullptr;
                if (box == nullptr) {
                    throw std::runtime_error("boxed ipm missing box_info in restoration sync");
                }
                outer_ipm.multiplier_.setZero();
                outer_ipm.d_multiplier_.setZero();
                outer_ipm.comp_.setZero();
                outer_ipm.v_.setConstant(-std::numeric_limits<scalar_t>::infinity());
                for (auto side : box_sides) {
                    auto &outer_pair = *outer_ipm.box_side_[side];
                    const auto &resto_side = overlay_data.elastic.side[side];
                    outer_pair.d_slack =
                        box->present_mask[side].select(resto_side.value[detail::slot_t].array() - outer_pair.slack.array(), scalar_t(0)).matrix();
                    outer_pair.d_multiplier =
                        box->present_mask[side]
                            .select((mu - outer_pair.multiplier.array() * outer_pair.d_slack.array()) /
                                        resto_side.value[detail::slot_t].array() -
                                        outer_pair.multiplier.array(),
                                    scalar_t(0))
                            .matrix();
                    outer_pair.slack =
                        box->present_mask[side].select(resto_side.value[detail::slot_t].array(), scalar_t(0)).matrix();
                    outer_pair.slack_backup = outer_pair.slack;
                    outer_pair.multiplier_backup = outer_pair.multiplier;
                    static_cast<solver::ipm_constr::approx_data::side_data &>(outer_pair).r_s =
                        box->present_mask[side].select(outer_pair.multiplier.array() * outer_pair.slack.array(), scalar_t(0)).matrix();
                    const scalar_t dual_sign = side == box_side::ub ? scalar_t(1) : scalar_t(-1);
                    outer_ipm.multiplier_.array() +=
                        dual_sign * box->present_mask[side].select(outer_pair.multiplier.array(), scalar_t(0));
                    outer_ipm.d_multiplier_.array() +=
                        dual_sign * box->present_mask[side].select(outer_pair.d_multiplier.array(), scalar_t(0));
                    outer_ipm.v_ =
                        box->present_mask[side]
                            .select(outer_ipm.v_.cwiseMax(resto_side.r_d).array(), outer_ipm.v_.array())
                            .matrix();
                    outer_ipm.comp_ =
                        box->present_mask[side]
                            .select((outer_pair.multiplier.array() * outer_pair.slack.array()).abs()
                                        .max(outer_ipm.comp_.array()),
                                    outer_ipm.comp_.array())
                            .matrix();
                }
            });
    }
}

} // namespace moto::solver::restoration
