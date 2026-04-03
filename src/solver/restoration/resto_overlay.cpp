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
size_t local_state_dim(const detail::eq_local_state &state) {
    return state.ns + state.nc;
}

size_t local_state_dim(const detail::ineq_local_state &state) {
    return state.nx + state.nu;
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
    for (auto *v : {&state.p, &state.p_backup, &state.d_p, &state.nu_p, &state.nu_p_backup, &state.d_nu_p,
                    &state.n, &state.n_backup, &state.d_n, &state.nu_n, &state.nu_n_backup, &state.d_nu_n,
                    &state.c_current, &state.r_c, &state.r_p, &state.r_n, &state.r_s_p, &state.r_s_n,
                    &state.combo_p, &state.combo_n, &state.b_c, &state.minv_diag, &state.minv_bc, &state.d_lambda,
                    &state.corrector_p, &state.corrector_n}) {
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
    for (auto *v : {&state.t, &state.t_backup, &state.d_t, &state.nu_t, &state.nu_t_backup, &state.d_nu_t,
                    &state.p, &state.p_backup, &state.d_p, &state.nu_p, &state.nu_p_backup, &state.d_nu_p,
                    &state.n, &state.n_backup, &state.d_n, &state.nu_n, &state.nu_n_backup, &state.d_nu_n,
                    &state.g_current, &state.r_d, &state.r_p, &state.r_n, &state.r_s_t, &state.r_s_p, &state.r_s_n,
                    &state.denom_t, &state.denom_p, &state.denom_n,
                    &state.combo_t, &state.combo_p, &state.combo_n, &state.b_d, &state.minv_diag, &state.minv_bd,
                    &state.corrector_t, &state.corrector_p, &state.corrector_n}) {
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
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model requires lambda_reg >= 0");
    }
    if (c.size() != lambda.size() ||
        c.size() != elastic.p.size() ||
        c.size() != elastic.n.size() ||
        c.size() != elastic.nu_p.size() ||
        c.size() != elastic.nu_n.size()) {
        throw std::runtime_error("compute_local_model size mismatch");
    }

    elastic.c_current = c;
    elastic.r_c.noalias() = c - elastic.p + elastic.n;
    elastic.r_p = vector::Constant(c.size(), rho) - lambda - elastic.nu_p;
    elastic.r_n = vector::Constant(c.size(), rho) + lambda - elastic.nu_n;
    if (mu_p_target != nullptr && mu_p_target->size() != c.size()) {
        throw std::runtime_error("compute_local_model mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != c.size()) {
        throw std::runtime_error("compute_local_model mu_n_target size mismatch");
    }
    for (Eigen::Index i = 0; i < c.size(); ++i) {
        const scalar_t mu_p = mu_p_target ? (*mu_p_target)(i) : mu_bar;
        const scalar_t mu_n = mu_n_target ? (*mu_n_target)(i) : mu_bar;
        elastic.r_s_p(i) = elastic.nu_p(i) * elastic.p(i) - mu_p;
        elastic.r_s_n(i) = elastic.nu_n(i) * elastic.n(i) - mu_n;
    }

    if (lambda_reg == 0.) {
        const auto denom_p = elastic.nu_p.array().max(eps);
        const auto denom_n = elastic.nu_n.array().max(eps);
        elastic.combo_p =
            (elastic.r_s_p.array() + elastic.p.array() * elastic.r_p.array()) / denom_p;
        elastic.combo_n =
            (elastic.r_s_n.array() + elastic.n.array() * elastic.r_n.array()) / denom_n;
        elastic.minv_diag =
            (elastic.p.array() / denom_p + elastic.n.array() / denom_n).inverse();
    } else {
        const auto denom_p = (elastic.p.array() + lambda_reg * elastic.nu_p.array()).max(eps);
        const auto denom_n = (elastic.n.array() + lambda_reg * elastic.nu_n.array()).max(eps);
        elastic.combo_p =
            (elastic.p.array() * elastic.r_p.array() + lambda_reg * elastic.r_s_p.array()) / denom_p;
        elastic.combo_n =
            (elastic.n.array() * elastic.r_n.array() + lambda_reg * elastic.r_s_n.array()) / denom_n;
        elastic.minv_diag =
            (elastic.p.array() / denom_p + elastic.n.array() / denom_n).inverse();
    }

    elastic.b_c = elastic.r_c + elastic.combo_p - elastic.combo_n;
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
    if (lambda_reg == 0.) {
        elastic.d_nu_p = elastic.r_p - elastic.d_lambda;
        elastic.d_nu_n = elastic.r_n + elastic.d_lambda;
        elastic.d_p =
            -(elastic.r_s_p.array() + elastic.p.array() * elastic.d_nu_p.array()) /
            elastic.nu_p.array().max(eps);
        elastic.d_n =
            -(elastic.r_s_n.array() + elastic.n.array() * elastic.d_nu_n.array()) /
            elastic.nu_n.array().max(eps);
    } else {
        const auto denom_p = (elastic.p.array() + lambda_reg * elastic.nu_p.array()).max(eps);
        const auto denom_n = (elastic.n.array() + lambda_reg * elastic.nu_n.array()).max(eps);
        elastic.d_nu_p =
            (elastic.nu_p.array() * (elastic.r_p.array() - elastic.d_lambda.array()) - elastic.r_s_p.array()) /
            denom_p;
        elastic.d_nu_n =
            (elastic.nu_n.array() * (elastic.r_n.array() + elastic.d_lambda.array()) - elastic.r_s_n.array()) /
            denom_n;
        elastic.d_p =
            elastic.d_lambda.array() + lambda_reg * elastic.d_nu_p.array() - elastic.r_p.array();
        elastic.d_n =
            -elastic.d_lambda.array() + lambda_reg * elastic.d_nu_n.array() - elastic.r_n.array();
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
    if (lambda_reg < 0.) {
        throw std::runtime_error("compute_local_model(ineq) requires lambda_reg >= 0");
    }
    if (g.size() != ineq.t.size() ||
        g.size() != ineq.nu_t.size() ||
        g.size() != ineq.p.size() ||
        g.size() != ineq.n.size()) {
        throw std::runtime_error("compute_local_model(ineq) size mismatch");
    }

    ineq.g_current = g;
    ineq.r_d = g + ineq.t - ineq.p + ineq.n;
    ineq.r_p = vector::Constant(g.size(), rho) - ineq.nu_t - ineq.nu_p;
    ineq.r_n = vector::Constant(g.size(), rho) + ineq.nu_t - ineq.nu_n;
    if (mu_t_target != nullptr && mu_t_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_t_target size mismatch");
    }
    if (mu_p_target != nullptr && mu_p_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_p_target size mismatch");
    }
    if (mu_n_target != nullptr && mu_n_target->size() != g.size()) {
        throw std::runtime_error("compute_local_model(ineq) mu_n_target size mismatch");
    }
    for (Eigen::Index i = 0; i < g.size(); ++i) {
        const scalar_t mu_t = mu_t_target ? (*mu_t_target)(i) : mu_bar;
        const scalar_t mu_p = mu_p_target ? (*mu_p_target)(i) : mu_bar;
        const scalar_t mu_n = mu_n_target ? (*mu_n_target)(i) : mu_bar;
        ineq.r_s_t(i) = ineq.nu_t(i) * ineq.t(i) - mu_t;
        ineq.r_s_p(i) = ineq.nu_p(i) * ineq.p(i) - mu_p;
        ineq.r_s_n(i) = ineq.nu_n(i) * ineq.n(i) - mu_n;
    }

    ineq.denom_t = ineq.nu_t.array().max(eps);
    ineq.denom_p = (ineq.p.array() + lambda_reg * ineq.nu_p.array()).max(eps);
    ineq.denom_n = (ineq.n.array() + lambda_reg * ineq.nu_n.array()).max(eps);

    ineq.combo_t = ineq.r_s_t.array() / ineq.denom_t.array();
    ineq.combo_p =
        (ineq.p.array() * ineq.r_p.array() + lambda_reg * ineq.r_s_p.array()) /
        ineq.denom_p.array();
    ineq.combo_n =
        (ineq.n.array() * ineq.r_n.array() + lambda_reg * ineq.r_s_n.array()) /
        ineq.denom_n.array();

    ineq.minv_diag =
        (ineq.t.array() / ineq.denom_t.array() +
         ineq.p.array() / ineq.denom_p.array() +
         ineq.n.array() / ineq.denom_n.array()).inverse();
    ineq.b_d = ineq.r_d - ineq.combo_t + ineq.combo_p - ineq.combo_n;
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
    ineq.d_nu_t = ineq.minv_diag.array() * (delta_g.array() + ineq.b_d.array());
    ineq.d_nu_p =
        (ineq.nu_p.array() * (ineq.r_p.array() - ineq.d_nu_t.array()) - ineq.r_s_p.array()) /
        ineq.denom_p.array();
    ineq.d_nu_n =
        (ineq.nu_n.array() * (ineq.r_n.array() + ineq.d_nu_t.array()) - ineq.r_s_n.array()) /
        ineq.denom_n.array();
    ineq.d_t =
        -(ineq.r_s_t.array() + ineq.t.array() * ineq.d_nu_t.array()) /
        ineq.denom_t.array();
    ineq.d_p =
        ineq.d_nu_t.array() + lambda_reg * ineq.d_nu_p.array() - ineq.r_p.array();
    ineq.d_n =
        -ineq.d_nu_t.array() + lambda_reg * ineq.d_nu_n.array() - ineq.r_n.array();
}

local_residual_summary resto_eq_elastic_constr::current_local_residuals(const detail::eq_local_state &elastic) {
    local_residual_summary out;
    if (local_state_dim(elastic) == 0) {
        return out;
    }
    out.inf_prim = elastic.r_c.cwiseAbs().maxCoeff();
    out.inf_stat = std::max(elastic.r_p.cwiseAbs().maxCoeff(), elastic.r_n.cwiseAbs().maxCoeff());
    out.inf_comp = std::max(elastic.r_s_p.cwiseAbs().maxCoeff(), elastic.r_s_n.cwiseAbs().maxCoeff());
    return out;
}

local_residual_summary resto_ineq_elastic_ipm_constr::current_local_residuals(const detail::ineq_local_state &ineq) {
    local_residual_summary out;
    if (local_state_dim(ineq) == 0) {
        return out;
    }
    out.inf_prim = ineq.r_d.cwiseAbs().maxCoeff();
    out.inf_stat = std::max({ineq.r_p.cwiseAbs().maxCoeff(), ineq.r_n.cwiseAbs().maxCoeff()});
    out.inf_comp = std::max({ineq.r_s_t.cwiseAbs().maxCoeff(),
                             ineq.r_s_p.cwiseAbs().maxCoeff(),
                             ineq.r_s_n.cwiseAbs().maxCoeff()});
    return out;
}

local_residual_summary resto_eq_elastic_constr::linearized_newton_residuals(const vector_const_ref &delta_c,
                                                                            const detail::eq_local_state &elastic,
                                                                            scalar_t lambda_reg) {
    local_residual_summary out;
    if (local_state_dim(elastic) == 0) {
        return out;
    }
    const vector res_c = delta_c - elastic.d_p + elastic.d_n + elastic.r_c;
    const vector res_p = elastic.d_p - elastic.d_lambda - lambda_reg * elastic.d_nu_p + elastic.r_p;
    const vector res_n = elastic.d_n + elastic.d_lambda - lambda_reg * elastic.d_nu_n + elastic.r_n;
    const vector res_sp =
        elastic.nu_p.cwiseProduct(elastic.d_p) + elastic.p.cwiseProduct(elastic.d_nu_p) + elastic.r_s_p;
    const vector res_sn =
        elastic.nu_n.cwiseProduct(elastic.d_n) + elastic.n.cwiseProduct(elastic.d_nu_n) + elastic.r_s_n;
    out.inf_prim = res_c.cwiseAbs().maxCoeff();
    out.inf_stat = std::max(res_p.cwiseAbs().maxCoeff(), res_n.cwiseAbs().maxCoeff());
    out.inf_comp = std::max(res_sp.cwiseAbs().maxCoeff(), res_sn.cwiseAbs().maxCoeff());
    return out;
}

local_residual_summary resto_ineq_elastic_ipm_constr::linearized_newton_residuals(const vector_const_ref &delta_g,
                                                                                   const detail::ineq_local_state &ineq,
                                                                                   scalar_t lambda_reg) {
    local_residual_summary out;
    if (local_state_dim(ineq) == 0) {
        return out;
    }
    const vector res_d = delta_g + ineq.d_t - ineq.d_p + ineq.d_n + ineq.r_d;
    const vector res_p = ineq.d_p - ineq.d_nu_t - lambda_reg * ineq.d_nu_p + ineq.r_p;
    const vector res_n = ineq.d_n + ineq.d_nu_t - lambda_reg * ineq.d_nu_n + ineq.r_n;
    const vector res_st =
        ineq.nu_t.cwiseProduct(ineq.d_t) + ineq.t.cwiseProduct(ineq.d_nu_t) + ineq.r_s_t;
    const vector res_sp =
        ineq.nu_p.cwiseProduct(ineq.d_p) + ineq.p.cwiseProduct(ineq.d_nu_p) + ineq.r_s_p;
    const vector res_sn =
        ineq.nu_n.cwiseProduct(ineq.d_n) + ineq.n.cwiseProduct(ineq.d_nu_n) + ineq.r_s_n;
    out.inf_prim = res_d.cwiseAbs().maxCoeff();
    out.inf_stat = std::max({res_p.cwiseAbs().maxCoeff(), res_n.cwiseAbs().maxCoeff()});
    out.inf_comp = std::max({res_st.cwiseAbs().maxCoeff(),
                             res_sp.cwiseAbs().maxCoeff(),
                             res_sn.cwiseAbs().maxCoeff()});
    return out;
}

namespace {

scalar_t max_abs_or_zero(const vector &v) {
    return v.size() > 0 ? v.cwiseAbs().maxCoeff() : scalar_t(0.);
}

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
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        for (const auto &[value, step, dual, dual_step] : {
                 std::tuple{&d.elastic.p, &d.elastic.d_p, &d.elastic.nu_p, &d.elastic.d_nu_p},
                 std::tuple{&d.elastic.n, &d.elastic.d_n, &d.elastic.nu_n, &d.elastic.d_nu_n},
             }) {
            if (value->size() == 0) {
                continue;
            }
            worker.n_ipm_cstr += static_cast<size_t>(value->size());
            worker.prev_aff_comp += dual->dot(*value);
            worker.post_aff_comp +=
                (*dual + alpha_dual * *dual_step).dot(*value + alpha_primal * *step);
        }
    } else {
        for (const auto &[value, step, dual, dual_step] : {
                 std::tuple{&d.elastic.t, &d.elastic.d_t, &d.elastic.nu_t, &d.elastic.d_nu_t},
                 std::tuple{&d.elastic.p, &d.elastic.d_p, &d.elastic.nu_p, &d.elastic.d_nu_p},
                 std::tuple{&d.elastic.n, &d.elastic.d_n, &d.elastic.nu_n, &d.elastic.d_nu_n},
             }) {
            if (value->size() == 0) {
                continue;
            }
            worker.n_ipm_cstr += static_cast<size_t>(value->size());
            worker.prev_aff_comp += dual->dot(*value);
            worker.post_aff_comp +=
                (*dual + alpha_dual * *dual_step).dot(*value + alpha_primal * *step);
        }
    }
    if constexpr (requires { d.elastic.corrector_t; d.elastic.d_nu_t; d.elastic.d_t; }) {
        d.elastic.corrector_t.array() =
            pair_dual_alpha * d.elastic.d_nu_t.array() * ls.alpha_primal * d.elastic.d_t.array();
    }
    d.elastic.corrector_p.array() =
        pair_dual_alpha * d.elastic.d_nu_p.array() * ls.alpha_primal * d.elastic.d_p.array();
    d.elastic.corrector_n.array() =
        pair_dual_alpha * d.elastic.d_nu_n.array() * ls.alpha_primal * d.elastic.d_n.array();
}

template <typename ApproxData>
void apply_corrector_pairs_like_ipm(ApproxData &d) {
    if (d.ipm_cfg == nullptr || local_state_dim(d.elastic) == 0) {
        return;
    }
    if (!d.ipm_cfg->ipm_accept_corrector()) {
        if constexpr (requires { d.elastic.corrector_t; }) {
            d.elastic.corrector_t.setZero();
        }
        d.elastic.corrector_p.setZero();
        d.elastic.corrector_n.setZero();
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
    } else if constexpr (requires { d.elastic.d_nu_t; }) {
        d.d_multiplier_ = d.elastic.d_nu_t;
    } else {
        d.d_multiplier_.setZero();
    }
}

template <typename ApproxData, typename InitFn>
void initialize_pair_overlay(ApproxData &d, InitFn &&init_fn) {
    for (Eigen::Index i = 0; i < d.base_residual.size(); ++i) {
        const auto init = init_fn(i);
        d.elastic.p(i) = init.p;
        d.elastic.n(i) = init.n;
        if constexpr (requires { init.nu_p; }) {
            d.elastic.nu_p(i) = init.nu_p;
            d.elastic.nu_n(i) = init.nu_n;
        } else {
            d.elastic.nu_p(i) = init.z_p;
            d.elastic.nu_n(i) = init.z_n;
        }
        if constexpr (requires { d.elastic.t; d.elastic.nu_t; }) {
            d.elastic.t(i) = init.t;
            d.elastic.nu_t(i) = init.nu_t;
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
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        positivity::apply_pair_step(d.elastic.p, d.elastic.d_p, ls.alpha_primal, d.elastic.nu_p, d.elastic.d_nu_p,
                                    pair_dual_alpha);
        positivity::apply_pair_step(d.elastic.n, d.elastic.d_n, ls.alpha_primal, d.elastic.nu_n, d.elastic.d_nu_n,
                                    pair_dual_alpha);
    } else {
        positivity::apply_pair_step(d.elastic.t, d.elastic.d_t, ls.alpha_primal, d.elastic.nu_t, d.elastic.d_nu_t,
                                    pair_dual_alpha);
        positivity::apply_pair_step(d.elastic.p, d.elastic.d_p, ls.alpha_primal, d.elastic.nu_p, d.elastic.d_nu_p,
                                    pair_dual_alpha);
        positivity::apply_pair_step(d.elastic.n, d.elastic.d_n, ls.alpha_primal, d.elastic.nu_n, d.elastic.d_nu_n,
                                    pair_dual_alpha);
    }
    if constexpr (requires { d.elastic.nu_t; }) {
        d.multiplier_ = d.elastic.nu_t;
    } else {
        d.multiplier_.noalias() += multiplier_dual_alpha * d.d_multiplier_;
    }
    if (d.ipm_cfg->ipm_accept_corrector()) {
        if constexpr (requires { d.elastic.t; d.elastic.nu_t; }) {
            d.elastic.t = d.elastic.t.array().max(1e-20);
            d.elastic.nu_t = d.elastic.nu_t.array().max(1e-20);
            d.multiplier_ = d.elastic.nu_t;
        }
        d.elastic.p = d.elastic.p.array().max(1e-20);
        d.elastic.n = d.elastic.n.array().max(1e-20);
        d.elastic.nu_p = d.elastic.nu_p.array().max(1e-20);
        d.elastic.nu_n = d.elastic.nu_n.array().max(1e-20);
    }
}

template <typename ApproxData>
void update_pair_ls_bounds_like_ipm(ApproxData &d,
                                    workspace_data *cfg,
                                    std::string_view kind,
                                    std::string_view name) {
    auto &ls = cfg->as<linesearch_config>();
    const scalar_t before_primal = ls.primal.alpha_max;
    scalar_t alpha_t = scalar_t(1.);
    if constexpr (requires { d.elastic.t; d.elastic.d_t; }) {
        alpha_t = alpha_candidate(d.elastic.t, d.elastic.d_t);
    }
    const scalar_t alpha_p = alpha_candidate(d.elastic.p, d.elastic.d_p);
    const scalar_t alpha_n = alpha_candidate(d.elastic.n, d.elastic.d_n);
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        positivity::update_pair_bounds(ls, d.elastic.p, d.elastic.d_p, d.elastic.nu_p, d.elastic.d_nu_p);
        positivity::update_pair_bounds(ls, d.elastic.n, d.elastic.d_n, d.elastic.nu_n, d.elastic.d_nu_n);
    } else {
        positivity::update_pair_bounds(ls, d.elastic.t, d.elastic.d_t, d.elastic.nu_t, d.elastic.d_nu_t);
        positivity::update_pair_bounds(ls, d.elastic.p, d.elastic.d_p, d.elastic.nu_p, d.elastic.d_nu_p);
        positivity::update_pair_bounds(ls, d.elastic.n, d.elastic.d_n, d.elastic.nu_n, d.elastic.d_nu_n);
    }
    if (resto_ls_debug_enabled() && ls.primal.alpha_max < before_primal) {
        fmt::print("    [resto-ls:{}:{}] alpha_t={:.3e} alpha_p={:.3e} (p={:.3e}, dp={:.3e}) alpha_n={:.3e} (n={:.3e}, dn={:.3e}) -> primal.alpha_max={:.3e}\n",
                   kind, name,
                   alpha_t,
                   alpha_p, d.elastic.p.minCoeff(), d.elastic.d_p.minCoeff(),
                   alpha_n, d.elastic.n.minCoeff(), d.elastic.d_n.minCoeff(),
                   ls.primal.alpha_max);
    }
}

template <typename ApproxData>
void backup_trial_pairs_like_ipm(ApproxData &d) {
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        positivity::backup_pair(d.elastic.p, d.elastic.p_backup, d.elastic.nu_p, d.elastic.nu_p_backup);
        positivity::backup_pair(d.elastic.n, d.elastic.n_backup, d.elastic.nu_n, d.elastic.nu_n_backup);
    } else {
        positivity::backup_pair(d.elastic.t, d.elastic.t_backup, d.elastic.nu_t, d.elastic.nu_t_backup);
        positivity::backup_pair(d.elastic.p, d.elastic.p_backup, d.elastic.nu_p, d.elastic.nu_p_backup);
        positivity::backup_pair(d.elastic.n, d.elastic.n_backup, d.elastic.nu_n, d.elastic.nu_n_backup);
    }
    d.multiplier_backup = d.multiplier_;
}

template <typename ApproxData>
void restore_trial_pairs_like_ipm(ApproxData &d) {
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        positivity::restore_pair(d.elastic.p, d.elastic.p_backup, d.elastic.nu_p, d.elastic.nu_p_backup);
        positivity::restore_pair(d.elastic.n, d.elastic.n_backup, d.elastic.nu_n, d.elastic.nu_n_backup);
    } else {
        positivity::restore_pair(d.elastic.t, d.elastic.t_backup, d.elastic.nu_t, d.elastic.nu_t_backup);
        positivity::restore_pair(d.elastic.p, d.elastic.p_backup, d.elastic.nu_p, d.elastic.nu_p_backup);
        positivity::restore_pair(d.elastic.n, d.elastic.n_backup, d.elastic.nu_n, d.elastic.nu_n_backup);
    }
    d.multiplier_ = d.multiplier_backup;
}

template <typename ApproxData>
scalar_t objective_penalty_from_pairs(const ApproxData &d) {
    return d.rho * (d.elastic.p.sum() + d.elastic.n.sum());
}

template <typename ApproxData>
scalar_t objective_penalty_dir_deriv_from_pairs(const ApproxData &d) {
    return d.rho * (d.elastic.d_p.sum() + d.elastic.d_n.sum());
}

template <typename ApproxData>
scalar_t search_penalty_from_pairs(const ApproxData &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        return d.ipm_cfg->mu *
               (d.elastic.p.array().max(scalar_t(1e-16)).log().sum() +
                d.elastic.n.array().max(scalar_t(1e-16)).log().sum());
    } else {
        return d.ipm_cfg->mu *
               (d.elastic.t.array().max(scalar_t(1e-16)).log().sum() +
                d.elastic.p.array().max(scalar_t(1e-16)).log().sum() +
                d.elastic.n.array().max(scalar_t(1e-16)).log().sum());
    }
}

template <typename ApproxData>
scalar_t search_penalty_dir_deriv_from_pairs(const ApproxData &d) {
    if (d.ipm_cfg == nullptr) {
        return 0.;
    }
    if constexpr (std::is_same_v<ApproxData, resto_eq_elastic_constr::approx_data>) {
        return d.ipm_cfg->mu *
               ((d.elastic.d_p.array() / d.elastic.p_backup.array().max(scalar_t(1e-16))).sum() +
                (d.elastic.d_n.array() / d.elastic.n_backup.array().max(scalar_t(1e-16))).sum());
    } else {
        return d.ipm_cfg->mu *
               ((d.elastic.d_t.array() / d.elastic.t_backup.array().max(scalar_t(1e-16))).sum() +
                (d.elastic.d_p.array() / d.elastic.p_backup.array().max(scalar_t(1e-16))).sum() +
                (d.elastic.d_n.array() / d.elastic.n_backup.array().max(scalar_t(1e-16))).sum());
    }
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
    d.v_ = d.base_residual - d.elastic.p + d.elastic.n;
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
    vector mu_p = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector_p;
    vector mu_n = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector_n;
    compute_local_model(d, d.ipm_cfg->mu, &mu_p, &mu_n);
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
    d.multiplier_ = d.elastic.nu_t;
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
        Eigen::Index i_rp = 0, i_rn = 0, i_rst = 0, i_rsp = 0, i_rsn = 0;
        const scalar_t max_rp = e.r_p.size() ? e.r_p.cwiseAbs().maxCoeff(&i_rp) : 0.;
        const scalar_t max_rn = e.r_n.size() ? e.r_n.cwiseAbs().maxCoeff(&i_rn) : 0.;
        const scalar_t max_rst = e.r_s_t.size() ? e.r_s_t.cwiseAbs().maxCoeff(&i_rst) : 0.;
        const scalar_t max_rsp = e.r_s_p.size() ? e.r_s_p.cwiseAbs().maxCoeff(&i_rsp) : 0.;
        const scalar_t max_rsn = e.r_s_n.size() ? e.r_s_n.cwiseAbs().maxCoeff(&i_rsn) : 0.;
        const bool p_dominates = max_rp >= max_rn;
        const Eigen::Index i_stat = p_dominates ? i_rp : i_rn;
        Eigen::Index i_comp = i_rst;
        const char *which_comp = "r_s_t";
        if (max_rsp >= max_rst && max_rsp >= max_rsn) {
            i_comp = i_rsp;
            which_comp = "r_s_p";
        } else if (max_rsn >= max_rst && max_rsn >= max_rsp) {
            i_comp = i_rsn;
            which_comp = "r_s_n";
        }
        const char *which_stat = p_dominates ? "r_p" : "r_n";
        const scalar_t stat_value = p_dominates ? e.r_p(i_stat) : e.r_n(i_stat);
        const scalar_t comp_value =
            std::string_view(which_comp) == "r_s_t" ? e.r_s_t(i_comp)
            : std::string_view(which_comp) == "r_s_p" ? e.r_s_p(i_comp)
                                                      : e.r_s_n(i_comp);
        fmt::print(
            "    [resto-local:{}] stat={:.3e} {}({})={:.3e} "
            "comp={:.3e} {}({})={:.3e} "
            "state[stat]=[t={:.3e} p={:.3e} n={:.3e} nu_t={:.3e} nu_p={:.3e} nu_n={:.3e}] "
            "state[comp]=[t={:.3e} p={:.3e} n={:.3e} nu_t={:.3e} nu_p={:.3e} nu_n={:.3e}]\n",
            data.func_.name(), summary.inf_stat, which_stat, i_stat, stat_value,
            summary.inf_comp, which_comp, i_comp, comp_value,
            e.t(i_stat), e.p(i_stat), e.n(i_stat), e.nu_t(i_stat), e.nu_p(i_stat), e.nu_n(i_stat),
            e.t(i_comp), e.p(i_comp), e.n(i_comp), e.nu_t(i_comp), e.nu_p(i_comp), e.nu_n(i_comp));
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
    d.v_ = d.base_residual + d.elastic.t - d.elastic.p + d.elastic.n;
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
        d.elastic.t(i) = init.t;
        d.elastic.p(i) = init.p;
        d.elastic.n(i) = init.n;
        d.elastic.nu_t(i) = init.nu_t;
        d.elastic.nu_p(i) = init.nu_p;
        d.elastic.nu_n(i) = init.nu_n;
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
    vector mu_t = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector_t;
    vector mu_p = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector_p;
    vector mu_n = vector::Constant(d.func_.dim(), d.ipm_cfg->mu) - d.elastic.corrector_n;
    compute_local_model(d, d.ipm_cfg->mu, &mu_t, &mu_p, &mu_n);
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
