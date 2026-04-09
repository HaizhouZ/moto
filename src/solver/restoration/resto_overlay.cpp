#include <moto/solver/restoration/resto_overlay.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <moto/ocp/impl/node_data.hpp>
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
                      &state.d_dual, &state.r_stat, &state.r_comp, &state.backsub_rhs, &state.corrector}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *v : {&state.base_residual, &state.r_c, &state.condensed_rhs,
                    &state.schur_inv_diag, &state.schur_rhs, &state.d_multiplier}) {
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
                      &state.d_dual, &state.r_comp, &state.denom, &state.backsub_rhs, &state.corrector}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *arr : {&state.r_stat}) {
        for (auto &v : *arr) {
            resize_zero(v, dim_eig);
        }
    }
    for (auto *v : {&state.base_residual, &state.r_d, &state.condensed_rhs,
                    &state.schur_inv_diag, &state.schur_rhs}) {
        resize_zero(*v, dim_eig);
    }
}

void resto_eq_elastic_constr::compute_local_model(detail::eq_local_state &elastic,
                                                  const vector_const_ref &c,
                                                  const vector_const_ref &lambda,
                                                  scalar_t rho,
                                                  scalar_t mu_bar,
                                                  const detail::elastic_pair_array<vector> *corrector) {
    const scalar_t eps = 1e-16;
    const Eigen::Index dim = c.size();
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
    for (auto slot : k_pair_slots) {
        elastic.r_comp[slot].array() = elastic.dual[slot].array() * elastic.value[slot].array() - mu_bar;
        if (corrector != nullptr) {
            elastic.r_comp[slot].array() += (*corrector)[slot].array();
        }
    }

    // Unregularized local Schur complement:
    // M_c = diag(p / z_p + n / z_n).
    elastic.schur_inv_diag.setZero(dim);
    for (auto slot : k_pair_slots) {
        const auto denom = elastic.dual[slot].array().max(eps);
        elastic.backsub_rhs[slot] =
            (elastic.value[slot].array() * elastic.r_stat[slot].array() + elastic.r_comp[slot].array()) / denom;
        elastic.schur_inv_diag.array() += elastic.value[slot].array() / denom;
    }
    elastic.schur_inv_diag = elastic.schur_inv_diag.array().inverse();

    elastic.condensed_rhs = elastic.r_c;
    for (size_t k : range(k_pair_slots.size())) {
        elastic.condensed_rhs.array() -= k_pair_signs[k] * elastic.backsub_rhs[k_pair_slots[k]].array();
    }
    elastic.schur_rhs = elastic.schur_inv_diag.array() * elastic.condensed_rhs.array();
}

void resto_ineq_elastic_ipm_constr::compute_local_model(detail::ineq_local_state &ineq,
                                                        const vector_const_ref &g,
                                                        scalar_t rho,
                                                        scalar_t mu_bar,
                                                        const detail::elastic_triplet_array<vector> *corrector) {
    const scalar_t eps = 1e-16;
    const Eigen::Index dim = g.size();
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
    for (auto slot : k_triplet_slots) {
        ineq.r_comp[slot].array() = ineq.dual[slot].array() * ineq.value[slot].array() - mu_bar;
        if (corrector != nullptr) {
            ineq.r_comp[slot].array() += (*corrector)[slot].array();
        }
    }

    // Unregularized local Schur complement:
    // M_d = diag(t / nu_t + p / nu_p + n / nu_n).
    ineq.schur_inv_diag.setZero(dim);
    for (auto slot : k_triplet_slots) {
        if (slot == detail::slot_t) {
            ineq.denom[slot] = ineq.dual[slot].array().max(eps);
            ineq.backsub_rhs[slot] = ineq.r_comp[slot].array() / ineq.denom[slot].array();
        } else {
            ineq.denom[slot] = ineq.dual[slot].array().max(eps);
            ineq.backsub_rhs[slot] =
                (ineq.value[slot].array() * ineq.r_stat[slot].array() + ineq.r_comp[slot].array()) /
                ineq.denom[slot].array();
        }
        ineq.schur_inv_diag.array() += ineq.value[slot].array() / ineq.denom[slot].array();
    }
    ineq.schur_inv_diag = ineq.schur_inv_diag.array().inverse();
    ineq.condensed_rhs = ineq.r_d;
    for (size_t k : range(k_triplet_slots.size())) {
        ineq.condensed_rhs.array() -= k_triplet_signs[k] * ineq.backsub_rhs[k_triplet_slots[k]].array();
    }
    ineq.schur_rhs = ineq.schur_inv_diag.array() * ineq.condensed_rhs.array();
}

local_residual_summary resto_eq_elastic_constr::current_local_residuals(const detail::eq_local_state &elastic) {
    local_residual_summary out;
    out.inf_prim = elastic.r_c.cwiseAbs().maxCoeff();
    for (auto slot : k_pair_slots) {
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(elastic.r_stat[slot]));
        out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(elastic.r_comp[slot]));
    }
    return out;
}

local_residual_summary resto_ineq_elastic_ipm_constr::current_local_residuals(const detail::ineq_local_state &ineq) {
    local_residual_summary out;
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
                                                                            const detail::eq_local_state &elastic) {
    local_residual_summary out;
    vector res_c = delta_c + elastic.r_c;
    for (size_t k : range(k_pair_slots.size())) {
        const auto slot = k_pair_slots[k];
        const scalar_t sign = k_pair_signs[k];
        res_c.array() += sign * elastic.d_value[slot].array();
        const vector res_stat = sign * elastic.d_multiplier - elastic.d_dual[slot] + elastic.r_stat[slot];
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
                                                                                  const detail::ineq_local_state &ineq) {
    local_residual_summary out;
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
    const vector res_p =
        -ineq.d_dual[detail::slot_t] - ineq.d_dual[detail::slot_p] + ineq.r_stat[detail::slot_p];
    const vector res_n =
        ineq.d_dual[detail::slot_t] - ineq.d_dual[detail::slot_n] + ineq.r_stat[detail::slot_n];
    out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_p));
    out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_n));
    out.inf_prim = res_d.cwiseAbs().maxCoeff();
    return out;
}

} // namespace moto::solver::restoration
