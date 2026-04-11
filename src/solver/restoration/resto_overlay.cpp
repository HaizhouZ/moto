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
    for (auto side : box_sides) {
        state.present_mask[side] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim_eig, side == box_side::ub);
    }
    for (auto side : box_sides) {
        auto &side_state = state.side[side];
        for (auto *arr : {&side_state.value, &side_state.value_backup, &side_state.d_value, &side_state.dual,
                          &side_state.dual_backup, &side_state.d_dual, &side_state.r_comp, &side_state.denom,
                          &side_state.backsub_rhs, &side_state.corrector}) {
            for (auto &v : *arr) {
                resize_zero(v, dim_eig);
            }
        }
        for (auto *arr : {&side_state.r_stat}) {
            for (auto &v : *arr) {
                resize_zero(v, dim_eig);
            }
        }
        for (auto *v : {&side_state.residual, &side_state.r_d, &side_state.condensed_rhs,
                        &side_state.schur_inv_diag, &side_state.schur_rhs}) {
            resize_zero(*v, dim_eig);
        }
    }
    for (auto *v : {&state.base_residual, &state.primal_view, &state.schur_rhs_net,
                    &state.schur_inv_diag_sum, &state.d_multiplier}) {
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
                                                        const ineq_constr::box_spec &box,
                                                        scalar_t rho,
                                                        scalar_t mu_bar,
                                                        const detail::elastic_side_array<detail::elastic_triplet_array<vector>> *corrector) {
    const scalar_t eps = 1e-16;
    const Eigen::Index dim = ineq.base_residual.size();
    ineq.primal_view.setConstant(dim, -std::numeric_limits<scalar_t>::infinity());
    ineq.schur_rhs_net.setZero(dim);
    ineq.schur_inv_diag_sum.setZero(dim);

    for (auto side : box_sides) {
        auto &side_state = ineq.side[side];
        for (auto slot : k_triplet_slots) {
            if (dim != side_state.value[slot].size() || dim != side_state.dual[slot].size()) {
                throw std::runtime_error("compute_local_model(ineq) size mismatch");
            }
        }

        side_state.r_d = side_state.residual;
        side_state.r_d.array() += side_state.value[detail::slot_t].array() -
                                  side_state.value[detail::slot_p].array() +
                                  side_state.value[detail::slot_n].array();
        side_state.r_d = box.present_mask[side].select(side_state.r_d.array(), scalar_t(0)).matrix();
        for (size_t k : range(k_pair_slots.size())) {
            const auto slot = k_pair_slots[k];
            const scalar_t sign = k_pair_signs[k];
            side_state.r_stat[slot] =
                vector::Constant(dim, rho) + sign * side_state.dual[detail::slot_t] - side_state.dual[slot];
            side_state.r_stat[slot] =
                box.present_mask[side].select(side_state.r_stat[slot].array(), scalar_t(0)).matrix();
        }
        for (auto slot : k_triplet_slots) {
            side_state.r_comp[slot].array() = side_state.dual[slot].array() * side_state.value[slot].array() - mu_bar;
            if (corrector != nullptr) {
                side_state.r_comp[slot].array() += (*corrector)[side][slot].array();
            }
            side_state.r_comp[slot] =
                box.present_mask[side].select(side_state.r_comp[slot].array(), scalar_t(0)).matrix();
        }

        side_state.schur_inv_diag.setZero(dim);
        for (auto slot : k_triplet_slots) {
            side_state.denom[slot] = side_state.dual[slot].array().max(eps);
            if (slot == detail::slot_t) {
                side_state.backsub_rhs[slot] =
                    side_state.r_comp[slot].array() / side_state.denom[slot].array();
            } else {
                side_state.backsub_rhs[slot] =
                    (side_state.value[slot].array() * side_state.r_stat[slot].array() +
                     side_state.r_comp[slot].array()) /
                    side_state.denom[slot].array();
            }
            side_state.schur_inv_diag.array() +=
                side_state.value[slot].array() / side_state.denom[slot].array();
        }
        side_state.schur_inv_diag = side_state.schur_inv_diag.array().inverse();
        side_state.condensed_rhs = side_state.r_d;
        side_state.condensed_rhs.array() -= side_state.backsub_rhs[detail::slot_t].array();
        side_state.condensed_rhs.array() += side_state.backsub_rhs[detail::slot_p].array();
        side_state.condensed_rhs.array() -= side_state.backsub_rhs[detail::slot_n].array();
        side_state.schur_rhs =
            box.present_mask[side].select(side_state.schur_inv_diag.array() * side_state.condensed_rhs.array(),
                                          scalar_t(0))
                .matrix();

        ineq.primal_view =
            box.present_mask[side]
                .select(ineq.primal_view.cwiseMax(side_state.r_d).array(), ineq.primal_view.array())
                .matrix();
        ineq.schur_rhs_net.array() +=
            side_jac_sign(side) * box.present_mask[side].select(side_state.schur_rhs.array(), scalar_t(0));
        ineq.schur_inv_diag_sum.array() +=
            box.present_mask[side].select(side_state.schur_inv_diag.array(), scalar_t(0));
    }
    if (ineq.primal_view.size() > 0) {
        const auto any_side = (box.present_mask[box_side::ub] || box.present_mask[box_side::lb]);
        ineq.primal_view = any_side.select(ineq.primal_view.array(), scalar_t(0)).matrix();
    }
    ineq.present_mask = box.present_mask;
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
    out.inf_prim = max_abs_or_zero(ineq.primal_view);
    for (auto side : box_sides) {
        const auto &side_state = ineq.side[side];
        out.inf_prim = std::max(out.inf_prim, max_abs_or_zero(side_state.r_d));
        for (auto slot : k_pair_slots) {
            out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(side_state.r_stat[slot]));
        }
        for (auto slot : k_triplet_slots) {
            out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(side_state.r_comp[slot]));
        }
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
    for (auto side : box_sides) {
        const auto &side_state = ineq.side[side];
        vector res_d = ineq.present_mask[side].select(side_jac_sign(side) * delta_g.array(), scalar_t(0)).matrix();
        res_d += side_state.r_d;
        res_d.array() += side_state.d_value[detail::slot_t].array() -
                         side_state.d_value[detail::slot_p].array() +
                         side_state.d_value[detail::slot_n].array();
        out.inf_prim = std::max(out.inf_prim, max_abs_or_zero(res_d));
        const vector res_p =
            -side_state.d_dual[detail::slot_t] - side_state.d_dual[detail::slot_p] + side_state.r_stat[detail::slot_p];
        const vector res_n =
            side_state.d_dual[detail::slot_t] - side_state.d_dual[detail::slot_n] + side_state.r_stat[detail::slot_n];
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_p));
        out.inf_stat = std::max(out.inf_stat, max_abs_or_zero(res_n));
        for (auto slot : k_triplet_slots) {
            const vector res_comp =
                side_state.dual[slot].cwiseProduct(side_state.d_value[slot]) +
                side_state.value[slot].cwiseProduct(side_state.d_dual[slot]) +
                side_state.r_comp[slot];
            out.inf_comp = std::max(out.inf_comp, max_abs_or_zero(res_comp));
        }
    }
    return out;
}

} // namespace moto::solver::restoration
