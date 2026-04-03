#include <moto/solver/restoration/resto_overlay.hpp>

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <tuple>
#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto::solver::restoration {

namespace {
constexpr std::array k_pair_slots{detail::slot_p, detail::slot_n};
constexpr std::array<scalar_t, 2> k_pair_signs{scalar_t(-1.), scalar_t(1.)};
constexpr std::array k_triplet_slots{detail::slot_t, detail::slot_p, detail::slot_n};
constexpr std::array<scalar_t, 3> k_triplet_signs{scalar_t(1.), scalar_t(-1.), scalar_t(1.)};

template <typename ApproxData>
constexpr auto active_slots() {
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        return k_pair_slots;
    } else {
        return k_triplet_slots;
    }
}

size_t local_state_dim(const detail::eq_local_state &state) {
    return state.ns + state.nc;
}

size_t local_state_dim(const detail::ineq_local_state &state) {
    return state.nx + state.nu;
}

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}
} // namespace

void resto_eq_elastic_constr::resize_local_state(detail::eq_local_state &state, size_t ns_dim, size_t nc_dim) {
    auto resize_zero = [](vector &v, Eigen::Index n) {
        v.resize(n);
        v.setZero();
    };

    state.ns = ns_dim;
    state.nc = nc_dim;
    const auto dim_eig = static_cast<Eigen::Index>(state.ns + state.nc);
    for (auto *arr : {&state.value, &state.value_backup, &state.d_value, &state.dual, &state.dual_backup,
                      &state.d_dual, &state.r_stat, &state.r_comp, &state.combo, &state.corrector}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *v : {&state.base_residual, &state.r_c, &state.b_c, &state.minv_diag, &state.minv_bc, &state.d_lambda}) {
        resize_zero(*v, dim_eig);
    }
}

void resto_ineq_elastic_ipm_constr::resize_local_state(detail::ineq_local_state &state, size_t nx_dim, size_t nu_dim) {
    auto resize_zero = [](vector &v, Eigen::Index n) {
        v.resize(n);
        v.setZero();
    };

    state.nx = nx_dim;
    state.nu = nu_dim;
    const auto dim_eig = static_cast<Eigen::Index>(state.nx + state.nu);
    for (auto *arr : {&state.value, &state.value_backup, &state.d_value, &state.dual, &state.dual_backup,
                      &state.d_dual, &state.r_comp, &state.denom, &state.combo, &state.corrector}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *arr : {&state.r_stat}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *v : {&state.base_residual, &state.r_d, &state.b_d, &state.minv_diag, &state.minv_bd}) {
        resize_zero(*v, dim_eig);
    }
}

elastic_init_pair resto_eq_elastic_constr::initialize_elastic_pair(scalar_t c, scalar_t rho, scalar_t mu_bar) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_pair requires mu_bar > 0");
    }

    const scalar_t disc = (mu_bar - rho * c) * (mu_bar - rho * c) + scalar_t(2.) * rho * mu_bar * c;
    const scalar_t sqrt_disc = std::sqrt(std::max(disc, scalar_t(0.)));
    const scalar_t n = (mu_bar - rho * c + sqrt_disc) / (scalar_t(2.) * rho);
    const scalar_t p = c + n;

    elastic_init_pair out;
    out.p = std::max(p, scalar_t(1e-16));
    out.n = std::max(n, scalar_t(1e-16));
    out.z_p = mu_bar / out.p;
    out.z_n = mu_bar / out.n;
    out.lambda = rho - out.z_p;
    const scalar_t a = out.z_p / out.p;
    const scalar_t b = out.z_n / out.n;
    out.weight = (a > 0. && b > 0.) ? (a * b) / (a + b) : scalar_t(0.);
    return out;
}

elastic_init_ineq_scalar resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(scalar_t g,
                                                                                        scalar_t rho,
                                                                                        scalar_t mu_bar,
                                                                                        scalar_t nu_t_init) {
    if (!(rho > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires rho > 0");
    }
    if (!(mu_bar > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires mu_bar > 0");
    }
    if (!(nu_t_init > 0.)) {
        throw std::runtime_error("initialize_elastic_ineq_scalar requires nu_t_init > 0");
    }

    elastic_init_ineq_scalar out;
    const scalar_t eps = scalar_t(1e-16);
    out.nu_t = std::max(nu_t_init, eps);
    out.t = mu_bar / out.nu_t;
    const auto pair = resto_eq_elastic_constr::initialize_elastic_pair(g + out.t, rho, mu_bar);
    out.p = pair.p;
    out.n = pair.n;
    out.nu_p = pair.z_p;
    out.nu_n = pair.z_n;
    return out;
}

void resto_eq_elastic_constr::compute_local_model(detail::eq_local_state &elastic,
                                                  const vector_const_ref &c,
                                                  const vector_const_ref &lambda,
                                                  scalar_t rho,
                                                  scalar_t mu_bar,
                                                  const vector *mu_p_target,
                                                  const vector *mu_n_target,
                                                  scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    const Eigen::Index dim = c.size();
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model requires lambda_reg >= 0");
    }
    if (dim != lambda.size()) {
        throw std::runtime_error("compute_local_model size mismatch");
    }
    for (auto slot : k_pair_slots) {
        if (dim != elastic.value[slot].size() || dim != elastic.dual[slot].size()) {
            throw std::runtime_error("compute_local_model size mismatch");
        }
    }

    elastic.base_residual = c;
    elastic.r_c = c;
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        elastic.r_c.array() += sign * elastic.value[slot].array();
        elastic.r_stat[slot] = vector::Constant(dim, rho) + sign * lambda - elastic.dual[slot];
    }
    if (mu_p_target != nullptr && mu_p_target->size() != dim) {
        throw std::runtime_error("compute_local_model mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != dim) {
        throw std::runtime_error("compute_local_model mu_n_target size mismatch");
    }
    if (mu_p_target == nullptr && mu_n_target == nullptr) {
        for (auto slot : k_pair_slots) {
            elastic.r_comp[slot].array() =
                elastic.dual[slot].array() * elastic.value[slot].array() - mu_bar;
        }
    } else {
        for (Eigen::Index i = 0; i < dim; ++i) {
            const std::array mu_targets{mu_p_target, mu_n_target};
            for (size_t k : range(k_pair_slots.size())) {
                const auto slot = k_pair_slots[k];
                const scalar_t mu_target = mu_targets[k] ? (*mu_targets[k])(i) : mu_bar;
                elastic.r_comp[slot](i) =
                    elastic.dual[slot](i) * elastic.value[slot](i) - mu_target;
            }
        }
    }

    elastic.minv_diag.setZero(dim);
    for (auto slot : k_pair_slots) {
        const auto denom = (elastic.value[slot].array() + lambda_reg * elastic.dual[slot].array()).max(eps);
        elastic.combo[slot] =
            (elastic.value[slot].array() * elastic.r_stat[slot].array() +
             lambda_reg * elastic.r_comp[slot].array()) / denom;
        elastic.minv_diag.array() += elastic.value[slot].array() / denom;
    }
    elastic.minv_diag = elastic.minv_diag.array().inverse();

    elastic.b_c = elastic.r_c;
    for (size_t k : range(k_pair_slots.size())) {
        elastic.b_c.array() -= k_pair_signs[k] * elastic.combo[k_pair_slots[k]].array();
    }
    elastic.minv_bc = elastic.minv_diag.array() * elastic.b_c.array();
}

void resto_eq_elastic_constr::compute_local_model(detail::eq_local_state &elastic,
                                                  const vector_const_ref &c,
                                                  const vector_const_ref &lambda,
                                                  scalar_t rho,
                                                  scalar_t mu_bar,
                                                  scalar_t lambda_reg) {
    compute_local_model(elastic, c, lambda, rho, mu_bar, nullptr, nullptr, lambda_reg);
}

void resto_eq_elastic_constr::recover_local_step(const vector_const_ref &delta_c,
                                                 detail::eq_local_state &elastic,
                                                 scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    if (delta_c.size() != local_state_dim(elastic)) {
        throw std::runtime_error("recover_local_step size mismatch");
    }
    elastic.d_lambda = elastic.minv_diag.array() * (delta_c.array() + elastic.b_c.array());
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        elastic.d_dual[slot] =
            (elastic.dual[slot].array() *
                 (elastic.r_stat[slot].array() + sign * elastic.d_lambda.array()) -
             elastic.r_comp[slot].array()) /
            (elastic.value[slot].array() + lambda_reg * elastic.dual[slot].array()).max(eps);
        elastic.d_value[slot] =
            -sign * elastic.d_lambda.array() + lambda_reg * elastic.d_dual[slot].array() -
            elastic.r_stat[slot].array();
    }
}

void resto_ineq_elastic_ipm_constr::compute_local_model(detail::ineq_local_state &ineq,
                                                        const vector_const_ref &g,
                                                        scalar_t rho,
                                                        scalar_t mu_bar,
                                                        const vector *mu_t_target,
                                                        const vector *mu_p_target,
                                                        const vector *mu_n_target,
                                                        scalar_t lambda_reg) {
    const scalar_t eps = 1e-16;
    const Eigen::Index dim = g.size();
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model(ineq) requires lambda_reg >= 0");
    }
    for (auto slot : k_triplet_slots) {
        if (dim != ineq.value[slot].size() || dim != ineq.dual[slot].size()) {
            throw std::runtime_error("compute_local_model(ineq) size mismatch");
        }
    }

    ineq.base_residual = g;
    ineq.r_d = g;
    for (size_t k : range(k_triplet_slots.size())) {
        const auto slot = k_triplet_slots[k];
        ineq.r_d.array() += k_triplet_signs[k] * ineq.value[slot].array();
    }
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        ineq.r_stat[slot] = vector::Constant(dim, rho) + sign * ineq.dual[detail::slot_t] - ineq.dual[slot];
    }
    if (mu_t_target != nullptr && mu_t_target->size() != dim) {
        throw std::runtime_error("compute_local_model(ineq) mu_t_target size mismatch");
    }
    if (mu_p_target != nullptr && mu_p_target->size() != dim) {
        throw std::runtime_error("compute_local_model(ineq) mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != dim) {
        throw std::runtime_error("compute_local_model(ineq) mu_n_target size mismatch");
    }
    const std::array mu_targets{mu_t_target, mu_p_target, mu_n_target};
    if (mu_t_target == nullptr && mu_p_target == nullptr && mu_n_target == nullptr) {
        for (auto slot : k_triplet_slots) {
            ineq.r_comp[slot].array() =
                ineq.dual[slot].array() * ineq.value[slot].array() - mu_bar;
        }
    } else {
        for (Eigen::Index i = 0; i < dim; ++i) {
            for (size_t k : range(k_triplet_slots.size())) {
                const auto slot = k_triplet_slots[k];
                const scalar_t mu_target = mu_targets[k] ? (*mu_targets[k])(i) : mu_bar;
                ineq.r_comp[slot](i) =
                    ineq.dual[slot](i) * ineq.value[slot](i) - mu_target;
            }
        }
    }

    ineq.minv_diag.setZero(dim);
    for (auto slot : k_triplet_slots) {
        if (slot == detail::slot_t) {
            ineq.denom[slot] = ineq.dual[slot].array().max(eps);
            ineq.combo[slot] = ineq.r_comp[slot].array() / ineq.denom[slot].array();
        } else {
            ineq.denom[slot] = (ineq.value[slot].array() + lambda_reg * ineq.dual[slot].array()).max(eps);
            ineq.combo[slot] =
                (ineq.value[slot].array() * ineq.r_stat[slot].array() +
                 lambda_reg * ineq.r_comp[slot].array()) /
                ineq.denom[slot].array();
        }
        ineq.minv_diag.array() += ineq.value[slot].array() / ineq.denom[slot].array();
    }
    ineq.minv_diag = ineq.minv_diag.array().inverse();
    ineq.b_d = ineq.r_d;
    for (size_t k : range(k_triplet_slots.size())) {
        ineq.b_d.array() -= k_triplet_signs[k] * ineq.combo[k_triplet_slots[k]].array();
    }
    ineq.minv_bd = ineq.minv_diag.array() * ineq.b_d.array();
}

void resto_ineq_elastic_ipm_constr::compute_local_model(detail::ineq_local_state &ineq,
                                                        const vector_const_ref &g,
                                                        scalar_t rho,
                                                        scalar_t mu_bar,
                                                        scalar_t lambda_reg) {
    compute_local_model(ineq, g, rho, mu_bar, nullptr, nullptr, nullptr, lambda_reg);
}

void resto_ineq_elastic_ipm_constr::recover_local_step(const vector_const_ref &delta_g,
                                                       detail::ineq_local_state &ineq,
                                                       scalar_t lambda_reg) {
    if (delta_g.size() != local_state_dim(ineq)) {
        throw std::runtime_error("recover_local_step(ineq) size mismatch");
    }
    if (lambda_reg < 0.) {
        throw std::runtime_error("recover_local_step(ineq) requires lambda_reg >= 0");
    }
    ineq.d_dual[detail::slot_t] = ineq.minv_diag.array() * (delta_g.array() + ineq.b_d.array());
    ineq.d_value[detail::slot_t] =
        -(ineq.r_comp[detail::slot_t].array() +
          ineq.value[detail::slot_t].array() * ineq.d_dual[detail::slot_t].array()) /
        ineq.denom[detail::slot_t].array();
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        ineq.d_dual[slot] =
            (ineq.dual[slot].array() *
                 (ineq.r_stat[slot].array() + sign * ineq.d_dual[detail::slot_t].array()) -
             ineq.r_comp[slot].array()) /
            ineq.denom[slot].array();
        ineq.d_value[slot] =
            -sign * ineq.d_dual[detail::slot_t].array() + lambda_reg * ineq.d_dual[slot].array() -
            ineq.r_stat[slot].array();
    }
}

local_residual_summary resto_eq_elastic_constr::current_local_residuals(const detail::eq_local_state &elastic) {
    local_residual_summary out;
    if (local_state_dim(elastic) == 0) {
        return out;
    }
    out.inf_prim = elastic.r_c.cwiseAbs().maxCoeff();
    for (auto slot : k_pair_slots) {
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(elastic.r_stat[slot]));
        out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(elastic.r_comp[slot]));
    }
    return out;
}

local_residual_summary resto_ineq_elastic_ipm_constr::current_local_residuals(const detail::ineq_local_state &ineq) {
    local_residual_summary out;
    if (local_state_dim(ineq) == 0) {
        return out;
    }
    out.inf_prim = ineq.r_d.cwiseAbs().maxCoeff();
    for (auto slot : k_pair_slots) {
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(ineq.r_stat[slot]));
    }
    for (auto slot : k_triplet_slots) {
        out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(ineq.r_comp[slot]));
    }
    return out;
}

local_residual_summary resto_eq_elastic_constr::linearized_newton_residuals(const vector_const_ref &delta_c,
                                                                            const detail::eq_local_state &elastic,
                                                                            scalar_t lambda_reg) {
    local_residual_summary out;
    if (local_state_dim(elastic) == 0) {
        return out;
    }
    vector res_c = delta_c + elastic.r_c;
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        res_c.array() += sign * elastic.d_value[slot].array();
        const vector res_stat =
            elastic.d_value[slot] + sign * elastic.d_lambda - lambda_reg * elastic.d_dual[slot] + elastic.r_stat[slot];
        const vector res_comp =
            elastic.dual[slot].cwiseProduct(elastic.d_value[slot]) +
            elastic.value[slot].cwiseProduct(elastic.d_dual[slot]) +
            elastic.r_comp[slot];
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_stat));
        out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(res_comp));
    }
    out.inf_prim = res_c.cwiseAbs().maxCoeff();
    return out;
}

local_residual_summary resto_ineq_elastic_ipm_constr::linearized_newton_residuals(const vector_const_ref &delta_g,
                                                                                   const detail::ineq_local_state &ineq,
                                                                                   scalar_t lambda_reg) {
    local_residual_summary out;
    if (local_state_dim(ineq) == 0) {
        return out;
    }
    vector res_d = delta_g + ineq.r_d;
    for (size_t k : range(k_triplet_slots.size())) {
        const auto slot = k_triplet_slots[k];
        res_d.array() += k_triplet_signs[k] * ineq.d_value[slot].array();
        const vector res_comp =
            ineq.dual[slot].cwiseProduct(ineq.d_value[slot]) +
            ineq.value[slot].cwiseProduct(ineq.d_dual[slot]) +
            ineq.r_comp[slot];
        out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(res_comp));
    }
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        const vector res_stat =
            ineq.d_value[slot] + sign * ineq.d_dual[detail::slot_t] - lambda_reg * ineq.d_dual[slot] + ineq.r_stat[slot];
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_stat));
    }
    out.inf_prim = res_d.cwiseAbs().maxCoeff();
    return out;
}

namespace {

bool resto_local_debug_enabled() {
    static const bool enabled = [] {
        if (const char *env = std::getenv("MOTO_RESTO_DEBUG_LOCAL")) {
            return std::string_view(env) != "0";
        }
        return false;
    }();
    return enabled;
}

bool resto_ls_debug_enabled() {
    static const bool enabled = [] {
        if (const char *env = std::getenv("MOTO_RESTO_DEBUG_LS")) {
            return std::string_view(env) != "0";
        }
        return false;
    }();
    return enabled;
}

scalar_t alpha_candidate(const vector &value,
                         const vector &step,
                         scalar_t fraction_to_boundary = scalar_t(0.995)) {
    return solver::positivity::alpha_max(value, step, fraction_to_boundary);
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

template <typename ApproxData>
void finalize_predictor_pairs_like_ipm(ApproxData &d,
                                       workspace_data *cfg,
                                       scalar_t pair_dual_alpha) {
    auto &worker = cfg->as<solver::ipm_config::worker_type>();
    auto &ls = cfg->as<linesearch_config>();
    assert(d.ipm_cfg != nullptr);
    assert(d.ipm_cfg->ipm_computing_affine_step() &&
           "ipm affine step computation not started but affine step is requested");
    const scalar_t alpha_primal = ls.alpha_primal;
    const scalar_t alpha_dual = pair_dual_alpha;
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
            pair_dual_alpha * d.elastic.d_dual[slot].array() * ls.alpha_primal *
            d.elastic.d_value[slot].array();
    }
}

template <typename ApproxData>
void apply_corrector_pairs_like_ipm(ApproxData &d) {
    if (d.ipm_cfg == nullptr || local_state_dim(d.elastic) == 0) {
        return;
    }
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        for (auto slot : active_slots<ApproxData>()) {
            d.elastic.corrector[slot].setZero();
        }
    }
}

template <typename ApproxData>
void propagate_pair_jacobian(ApproxData &d, const vector &minv_b) {
    size_t arg_idx = 0;
    for (auto &jac : d.jac_) {
        if (jac.size() != 0) {
            d.lag_jac_corr_[arg_idx].noalias() += minv_b.transpose() * jac;
        }
        ++arg_idx;
    }
}

template <typename ApproxData>
void propagate_pair_hessian(ApproxData &d) {
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

template <typename ApproxData>
void finalize_pair_newton_step(ApproxData &d) {
    if (local_state_dim(d.elastic) == 0) {
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
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        resto_eq_elastic_constr::recover_local_step(delta, d.elastic, d.lambda_reg);
    } else {
        resto_ineq_elastic_ipm_constr::recover_local_step(delta, d.elastic, d.lambda_reg);
    }
    if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
        if (resto_local_debug_enabled() && local_state_dim(d.elastic) > 0) {
            const auto summary =
                resto_ineq_elastic_ipm_constr::linearized_newton_residuals(delta, d.elastic, d.lambda_reg);
            fmt::print("    [resto-newton:{}] linres prim={:.3e} stat={:.3e} comp={:.3e}\n",
                       d.func_.name(), summary.inf_prim, summary.inf_stat, summary.inf_comp);
        }
    }
    if constexpr (requires { d.elastic.d_lambda; }) {
        d.d_multiplier_ = d.elastic.d_lambda;
    } else if constexpr (std::is_same_v<ApproxData, resto_ineq_elastic_ipm_constr::approx_data>) {
        d.d_multiplier_ = d.elastic.d_dual[detail::slot_t];
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
    const scalar_t before_primal = ls.primal.alpha_max;
    std::array<scalar_t, 3> alpha_by_slot{scalar_t(1.), scalar_t(1.), scalar_t(1.)};
    for (auto slot : active_slots<ApproxData>()) {
        alpha_by_slot[static_cast<size_t>(slot)] = alpha_candidate(d.elastic.value[slot], d.elastic.d_value[slot]);
        positivity::update_pair_bounds(ls, d.elastic.value[slot], d.elastic.d_value[slot],
                                       d.elastic.dual[slot], d.elastic.d_dual[slot]);
    }
    if (resto_ls_debug_enabled() && ls.primal.alpha_max < before_primal) {
        fmt::print("    [resto-ls:{}:{}] alpha_t={:.3e} alpha_p={:.3e} (p={:.3e}, dp={:.3e}) alpha_n={:.3e} (n={:.3e}, dn={:.3e}) -> primal.alpha_max={:.3e}\n",
                   kind, name,
                   alpha_by_slot[detail::slot_t],
                   alpha_by_slot[detail::slot_p],
                   d.elastic.value[detail::slot_p].minCoeff(), d.elastic.d_value[detail::slot_p].minCoeff(),
                   alpha_by_slot[detail::slot_n],
                   d.elastic.value[detail::slot_n].minCoeff(), d.elastic.d_value[detail::slot_n].minCoeff(),
                   ls.primal.alpha_max);
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
    return d.rho * sum;
}

template <typename ApproxData>
scalar_t objective_penalty_dir_deriv_from_pairs(const ApproxData &d) {
    scalar_t sum = 0.;
    for (auto slot : k_pair_slots) {
        sum += d.elastic.d_value[slot].sum();
    }
    return d.rho * sum;
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
                d.elastic.value_backup[slot].array().max(scalar_t(1e-16))).sum();
    }
    return d.ipm_cfg->mu * sum;
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

void resto_eq_elastic_constr::compute_local_model(approx_data &d,
                                                  scalar_t mu_bar,
                                                  const vector *mu_p_target,
                                                  const vector *mu_n_target) const {
    if (local_state_dim(d.elastic) == 0 || d.ipm_cfg == nullptr) {
        return;
    }
    resto_eq_elastic_constr::compute_local_model(d.elastic,
                                                 d.base_residual,
                                                 d.multiplier_,
                                                 d.rho,
                                                 mu_bar,
                                                 mu_p_target,
                                                 mu_n_target,
                                                 d.lambda_reg);
}

local_residual_summary resto_eq_elastic_constr::current_local_residuals(const approx_data &data) const {
    auto elastic = data.elastic;
    if (local_state_dim(elastic) == 0 || data.ipm_cfg == nullptr) {
        return resto_eq_elastic_constr::current_local_residuals(elastic);
    }
    resto_eq_elastic_constr::compute_local_model(elastic,
                                                 data.base_residual,
                                                 data.multiplier_,
                                                 data.rho,
                                                 data.ipm_cfg->mu,
                                                 data.lambda_reg);
    return resto_eq_elastic_constr::current_local_residuals(elastic);
}

void resto_eq_elastic_constr::value_impl(func_approx_data &data) const {
    forward_source_value(source_, data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    if (local_state_dim(d.elastic) == 0) {
        d.v_ = d.base_residual;
        return;
    }
    d.v_ = d.base_residual;
    for (size_t k : range(k_pair_slots.size())) {
        d.v_.array() += k_pair_signs[k] * d.elastic.value[k_pair_slots[k]].array();
    }
    compute_local_model(d, d.ipm_cfg->mu);
}

void resto_eq_elastic_constr::jacobian_impl(func_approx_data &data) const {
    forward_source_jacobian(source_, data);
    auto &d = data.as<approx_data>();
    compute_local_model(d, d.ipm_cfg->mu);
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || local_state_dim(d.elastic) == 0) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    compute_local_model(d, target_mu);
    propagate_jacobian(d);
}

void resto_eq_elastic_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        forward_source_hessian(source_, data);
    }
    auto &d = data.as<approx_data>();
    compute_local_model(d, d.ipm_cfg->mu);
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || local_state_dim(d.elastic) == 0) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    compute_local_model(d, target_mu);
    propagate_hessian(d);
}

void resto_eq_elastic_constr::propagate_jacobian(func_approx_data &data) const {
    propagate_pair_jacobian(data.as<approx_data>(), data.as<approx_data>().elastic.minv_bc);
}

void resto_eq_elastic_constr::propagate_hessian(func_approx_data &data) const {
    propagate_pair_hessian(data.as<approx_data>());
}

void resto_eq_elastic_constr::propagate_res_stats(func_approx_data &) const {}

void resto_eq_elastic_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    resto_eq_elastic_constr::resize_local_state(d.elastic, 0, d.func_.dim());
    initialize_pair_overlay(d, [&](Eigen::Index i) {
        return resto_eq_elastic_constr::initialize_elastic_pair(d.base_residual(i), d.rho, d.ipm_cfg->mu);
    });
    compute_local_model(d, d.ipm_cfg->mu);
}

void resto_eq_elastic_constr::finalize_newton_step(data_map_t &data) const {
    finalize_pair_newton_step(data.as<approx_data>());
}

void resto_eq_elastic_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    finalize_predictor_pairs_like_ipm(d, cfg, cfg->as<linesearch_config>().dual_alpha_for_ineq());
}

void resto_eq_elastic_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    apply_corrector_pairs_like_ipm(d);
    detail::elastic_pair_array<vector> mu_target;
    for (auto slot : k_pair_slots) {
        mu_target[slot] = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector[slot];
    }
    compute_local_model(d, d.ipm_cfg->mu, &mu_target[detail::slot_p], &mu_target[detail::slot_n]);
    propagate_jacobian(d);
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
    return current_local_residuals(static_cast<const approx_data &>(data)).inf_stat;
}

scalar_t resto_eq_elastic_constr::local_comp_residual_inf(const func_approx_data &data) const {
    return current_local_residuals(static_cast<const approx_data &>(data)).inf_comp;
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

void resto_ineq_elastic_ipm_constr::compute_local_model(approx_data &d,
                                                        scalar_t mu_bar,
                                                        const vector *mu_t_target,
                                                        const vector *mu_p_target,
                                                        const vector *mu_n_target) const {
    if (local_state_dim(d.elastic) == 0 || d.ipm_cfg == nullptr) {
        return;
    }
    resto_ineq_elastic_ipm_constr::compute_local_model(d.elastic,
                                                       d.base_residual,
                                                       d.rho,
                                                       mu_bar,
                                                       mu_t_target,
                                                       mu_p_target,
                                                       mu_n_target,
                                                       d.lambda_reg);
    d.multiplier_ = d.elastic.dual[detail::slot_t];
}

local_residual_summary resto_ineq_elastic_ipm_constr::current_local_residuals(const approx_data &data) const {
    auto elastic = data.elastic;
    if (local_state_dim(elastic) != 0 && data.ipm_cfg != nullptr) {
        resto_ineq_elastic_ipm_constr::compute_local_model(elastic,
                                                           data.base_residual,
                                                           data.rho,
                                                           data.ipm_cfg->mu,
                                                           data.lambda_reg);
    }
    const auto summary = resto_ineq_elastic_ipm_constr::current_local_residuals(elastic);
    if (resto_local_debug_enabled() && (summary.inf_stat > 1e-6 || summary.inf_comp > 1e-6)) {
        const auto &e = elastic;
        constexpr std::array stat_labels{"r_p", "r_n"};
        constexpr std::array comp_labels{"r_s_t", "r_s_p", "r_s_n"};
        std::array<Eigen::Index, 2> stat_index{};
        std::array<Eigen::Index, 3> comp_index{};
        std::array<scalar_t, 2> stat_max{};
        std::array<scalar_t, 3> comp_max{};
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            stat_max[k] = e.r_stat[slot].size() ? e.r_stat[slot].cwiseAbs().maxCoeff(&stat_index[k]) : 0.;
        }
        for (size_t k : range(k_triplet_slots.size())) {
            const auto slot = k_triplet_slots[k];
            comp_max[k] = e.r_comp[slot].size() ? e.r_comp[slot].cwiseAbs().maxCoeff(&comp_index[k]) : 0.;
        }
        const size_t stat_k = stat_max[0] >= stat_max[1] ? 0 : 1;
        size_t comp_k = 0;
        if (comp_max[1] >= comp_max[0] && comp_max[1] >= comp_max[2]) {
            comp_k = 1;
        } else if (comp_max[2] >= comp_max[0] && comp_max[2] >= comp_max[1]) {
            comp_k = 2;
        }
        const auto stat_slot = k_pair_slots[stat_k];
        const auto comp_slot = k_triplet_slots[comp_k];
        const Eigen::Index i_stat = stat_index[stat_k];
        const Eigen::Index i_comp = comp_index[comp_k];
        fmt::print(
            "    [resto-local:{}] stat={:.3e} {}({})={:.3e} "
            "comp={:.3e} {}({})={:.3e} "
            "state[stat]=[t={:.3e} p={:.3e} n={:.3e} nu_t={:.3e} nu_p={:.3e} nu_n={:.3e}] "
            "state[comp]=[t={:.3e} p={:.3e} n={:.3e} nu_t={:.3e} nu_p={:.3e} nu_n={:.3e}]\n",
            data.func_.name(), summary.inf_stat, stat_labels[stat_k], i_stat, e.r_stat[stat_slot](i_stat),
            summary.inf_comp, comp_labels[comp_k], i_comp, e.r_comp[comp_slot](i_comp),
            e.value[detail::slot_t](i_stat), e.value[detail::slot_p](i_stat), e.value[detail::slot_n](i_stat),
            e.dual[detail::slot_t](i_stat), e.dual[detail::slot_p](i_stat), e.dual[detail::slot_n](i_stat),
            e.value[detail::slot_t](i_comp), e.value[detail::slot_p](i_comp), e.value[detail::slot_n](i_comp),
            e.dual[detail::slot_t](i_comp), e.dual[detail::slot_p](i_comp), e.dual[detail::slot_n](i_comp));
    }
    return summary;
}

void resto_ineq_elastic_ipm_constr::value_impl(func_approx_data &data) const {
    forward_source_value(source_, data);
    auto &d = data.as<approx_data>();
    d.base_residual = d.v_;
    // Local restoration complementarity is summarized directly from the wrapper's
    // current local model. Keep lag_data::comp_ neutral so public stage comp
    // statistics are not polluted by restoration-only bookkeeping.
    d.comp_.setZero();
    if (local_state_dim(d.elastic) == 0) {
        d.v_ = d.base_residual;
        return;
    }
    d.v_ = d.base_residual;
    for (size_t k : range(k_triplet_slots.size())) {
        d.v_.array() += k_triplet_signs[k] * d.elastic.value[k_triplet_slots[k]].array();
    }
    compute_local_model(d, d.ipm_cfg->mu);
}

void resto_ineq_elastic_ipm_constr::jacobian_impl(func_approx_data &data) const {
    forward_source_jacobian(source_, data);
    auto &d = data.as<approx_data>();
    compute_local_model(d, d.ipm_cfg->mu);
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || local_state_dim(d.elastic) == 0) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    compute_local_model(d, target_mu);
    propagate_jacobian(d);
}

void resto_ineq_elastic_ipm_constr::hessian_impl(func_approx_data &data) const {
    if (source_->order() >= approx_order::second) {
        forward_source_hessian(source_, data);
    }
    auto &d = data.as<approx_data>();
    compute_local_model(d, d.ipm_cfg->mu);
    if (d.ipm_cfg == nullptr || d.ipm_cfg->disable_corrections || local_state_dim(d.elastic) == 0) {
        return;
    }
    const scalar_t target_mu = d.ipm_cfg->ipm_enable_affine_step() ? scalar_t(0.) : d.ipm_cfg->mu;
    compute_local_model(d, target_mu);
    propagate_hessian(d);
}

void resto_ineq_elastic_ipm_constr::propagate_jacobian(func_approx_data &data) const {
    propagate_pair_jacobian(data.as<approx_data>(), data.as<approx_data>().elastic.minv_bd);
}

void resto_ineq_elastic_ipm_constr::propagate_hessian(func_approx_data &data) const {
    propagate_pair_hessian(data.as<approx_data>());
}

void resto_ineq_elastic_ipm_constr::propagate_res_stats(func_approx_data &) const {}

void resto_ineq_elastic_ipm_constr::initialize(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    resto_ineq_elastic_ipm_constr::resize_local_state(d.elastic, d.func_.dim(), 0);
    const scalar_t eps = scalar_t(1e-16);
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const scalar_t nu_t0 = std::max(std::min(d.rho, d.multiplier_(i)), eps);
        const auto init =
            resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(d.base_residual(i), d.rho, d.ipm_cfg->mu, nu_t0);
        d.elastic.value[detail::slot_t](i) = init.t;
        d.elastic.dual[detail::slot_t](i) = init.nu_t;
        const std::array pair_values{init.p, init.n};
        const std::array pair_duals{init.nu_p, init.nu_n};
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            d.elastic.value[slot](i) = pair_values[k];
            d.elastic.dual[slot](i) = pair_duals[k];
        }
        d.multiplier_(i) = init.nu_t;
    }
    d.multiplier_backup = d.multiplier_;
    compute_local_model(d, d.ipm_cfg->mu);
}

void resto_ineq_elastic_ipm_constr::finalize_newton_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    finalize_pair_newton_step(d);
}

void resto_ineq_elastic_ipm_constr::finalize_predictor_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    finalize_predictor_pairs_like_ipm(d, cfg, cfg->as<linesearch_config>().dual_alpha_for_ineq());
}

void resto_ineq_elastic_ipm_constr::apply_corrector_step(data_map_t &data) const {
    auto &d = data.as<approx_data>();
    apply_corrector_pairs_like_ipm(d);
    detail::elastic_triplet_array<vector> mu_target;
    for (auto slot : k_triplet_slots) {
        mu_target[slot] = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector[slot];
    }
    compute_local_model(d, d.ipm_cfg->mu,
                        &mu_target[detail::slot_t],
                        &mu_target[detail::slot_p],
                        &mu_target[detail::slot_n]);
    propagate_jacobian(d);
}

void resto_ineq_elastic_ipm_constr::apply_affine_step(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    const scalar_t alpha_dual = cfg->as<linesearch_config>().dual_alpha_for_ineq();
    apply_affine_pairs_like_ipm(d, cfg, alpha_dual, alpha_dual);
}

void resto_ineq_elastic_ipm_constr::update_ls_bounds(data_map_t &data, workspace_data *cfg) const {
    auto &d = data.as<approx_data>();
    update_pair_ls_bounds_like_ipm(d, cfg, "ineq", name());
}

void resto_ineq_elastic_ipm_constr::backup_trial_state(data_map_t &data) const {
    backup_trial_pairs_like_ipm(data.as<approx_data>());
}

void resto_ineq_elastic_ipm_constr::restore_trial_state(data_map_t &data) const {
    restore_trial_pairs_like_ipm(data.as<approx_data>());
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty(const func_approx_data &data) const {
    return objective_penalty_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_ineq_elastic_ipm_constr::objective_penalty_dir_deriv(const func_approx_data &data) const {
    return objective_penalty_dir_deriv_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty(const func_approx_data &data) const {
    return search_penalty_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_ineq_elastic_ipm_constr::search_penalty_dir_deriv(const func_approx_data &data) const {
    return search_penalty_dir_deriv_from_pairs(static_cast<const approx_data &>(data));
}

scalar_t resto_ineq_elastic_ipm_constr::local_stat_residual_inf(const func_approx_data &data) const {
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

    if (settings.rho_ineq > scalar_t(0.)) {
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
