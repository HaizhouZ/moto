#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <cmath>

#include <moto/ocp/impl/node_data.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/solver/ipm/positivity_step.hpp>
#include <moto/solver/ineq_soft.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/solver/restoration/resto_overlay.hpp>

using namespace moto;
using namespace moto::solver;
using namespace moto::solver::restoration;

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

bool approx_zero(const vector &v, scalar_t tol = 1e-9) {
    return v.size() == 0 || v.cwiseAbs().maxCoeff() < tol;
}

bool approx_scalar(scalar_t a, scalar_t b, scalar_t tol = 1e-12) {
    return std::abs(a - b) < tol;
}

vector gather_entries(const vector &v, std::initializer_list<Eigen::Index> idxs) {
    vector out(static_cast<Eigen::Index>(idxs.size()));
    Eigen::Index k = 0;
    for (Eigen::Index idx : idxs) {
        out(k++) = v(idx);
    }
    return out;
}

template <typename State>
vector &slot_value(State &state, detail::elastic_slot_t slot) {
    return state.value[slot];
}

vector &slot_value(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].value[slot];
}

template <typename State>
const vector &slot_value(const State &state, detail::elastic_slot_t slot) {
    return state.value[slot];
}

const vector &slot_value(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].value[slot];
}

template <typename State>
vector &slot_value_backup(State &state, detail::elastic_slot_t slot) {
    return state.value_backup[slot];
}

vector &slot_value_backup(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].value_backup[slot];
}

template <typename State>
vector &slot_step(State &state, detail::elastic_slot_t slot) {
    return state.d_value[slot];
}

vector &slot_step(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].d_value[slot];
}

template <typename State>
const vector &slot_step(const State &state, detail::elastic_slot_t slot) {
    return state.d_value[slot];
}

const vector &slot_step(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].d_value[slot];
}

template <typename State>
vector &slot_dual(State &state, detail::elastic_slot_t slot) {
    return state.dual[slot];
}

vector &slot_dual(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].dual[slot];
}

template <typename State>
const vector &slot_dual(const State &state, detail::elastic_slot_t slot) {
    return state.dual[slot];
}

const vector &slot_dual(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].dual[slot];
}

template <typename State>
vector &slot_dual_backup(State &state, detail::elastic_slot_t slot) {
    return state.dual_backup[slot];
}

vector &slot_dual_backup(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].dual_backup[slot];
}

template <typename State>
vector &slot_dual_step(State &state, detail::elastic_slot_t slot) {
    return state.d_dual[slot];
}

vector &slot_dual_step(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].d_dual[slot];
}

template <typename State>
const vector &slot_dual_step(const State &state, detail::elastic_slot_t slot) {
    return state.d_dual[slot];
}

const vector &slot_dual_step(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].d_dual[slot];
}

template <typename State>
vector &slot_r_stat(State &state, detail::elastic_slot_t slot) {
    return state.r_stat[slot];
}

vector &slot_r_stat(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].r_stat[slot];
}

template <typename State>
const vector &slot_r_stat(const State &state, detail::elastic_slot_t slot) {
    return state.r_stat[slot];
}

const vector &slot_r_stat(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].r_stat[slot];
}

template <typename State>
vector &slot_r_comp(State &state, detail::elastic_slot_t slot) {
    return state.r_comp[slot];
}

vector &slot_r_comp(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].r_comp[slot];
}

template <typename State>
const vector &slot_r_comp(const State &state, detail::elastic_slot_t slot) {
    return state.r_comp[slot];
}

const vector &slot_r_comp(const detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].r_comp[slot];
}

template <typename State>
vector &slot_denom(State &state, detail::elastic_slot_t slot) {
    return state.denom[slot];
}

vector &slot_denom(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].denom[slot];
}

template <typename State>
vector &slot_backsub_rhs(State &state, detail::elastic_slot_t slot) {
    return state.backsub_rhs[slot];
}

vector &slot_backsub_rhs(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].backsub_rhs[slot];
}

template <typename State>
vector &slot_corrector(State &state, detail::elastic_slot_t slot) {
    return state.corrector[slot];
}

vector &slot_corrector(detail::ineq_local_state &state, detail::elastic_slot_t slot, box_side::side_t side = box_side::ub) {
    return state.side[side].corrector[slot];
}

ineq_constr::box_spec make_half_box_spec(Eigen::Index dim) {
    ineq_constr::box_spec spec;
    spec.base_dim = static_cast<size_t>(dim);
    spec.present_mask[box_side::ub] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim, true);
    spec.present_mask[box_side::lb] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim, false);
    spec.has_side[box_side::ub] = true;
    spec.has_side[box_side::lb] = false;
    return spec;
}

void resize_eq_state(restoration::detail::eq_local_state &state, size_t ns_dim, size_t nc_dim) {
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

void resize_ineq_state(restoration::detail::ineq_local_state &state, size_t nx_dim, size_t nu_dim) {
    auto resize_zero = [](vector &v, Eigen::Index n) {
        v.resize(n);
        v.setZero();
    };
    state.nx = nx_dim;
    state.nu = nu_dim;
    const auto dim_eig = static_cast<Eigen::Index>(state.nx + state.nu);
    state.present_mask[box_side::ub] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim_eig, true);
    state.present_mask[box_side::lb] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(dim_eig, false);
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

void recover_eq_local_step(const vector_const_ref &delta_c, restoration::detail::eq_local_state &eq) {
    const scalar_t eps = 1e-16;
    const std::array slots{detail::slot_p, detail::slot_n};
    const std::array<scalar_t, 2> signs{scalar_t(-1.), scalar_t(1.)};
    eq.d_multiplier = eq.schur_inv_diag.array() * (delta_c.array() + eq.condensed_rhs.array());
    for (size_t k : range(slots.size())) {
        const auto slot = slots[k];
        const scalar_t sign = signs[k];
        const auto scale = eq.value[slot].array() / eq.dual[slot].array().max(eps);
        eq.d_value[slot] = -sign * scale * eq.d_multiplier.array() - eq.backsub_rhs[slot].array();
        eq.d_dual[slot] = eq.r_stat[slot].array() + sign * eq.d_multiplier.array();
    }
}

void recover_ineq_local_step(const vector_const_ref &delta_g, restoration::detail::ineq_local_state &iq) {
    auto &side = iq.side[box_side::ub];
    const std::array slots{detail::slot_p, detail::slot_n};
    const std::array<scalar_t, 2> signs{scalar_t(-1.), scalar_t(1.)};
    side.d_dual[detail::slot_t] = side.schur_inv_diag.array() * (delta_g.array() + side.condensed_rhs.array());
    side.d_value[detail::slot_t] =
        -(side.value[detail::slot_t].array() / side.dual[detail::slot_t].array()) *
            side.d_dual[detail::slot_t].array() -
        side.backsub_rhs[detail::slot_t].array();
    for (size_t k : range(slots.size())) {
        const auto slot = slots[k];
        const scalar_t sign = signs[k];
        side.d_value[slot] =
            -sign * (side.value[slot].array() / side.dual[slot].array()) * side.d_dual[detail::slot_t].array() -
            side.backsub_rhs[slot].array();
        side.d_dual[slot] = side.r_stat[slot].array() + sign * side.d_dual[detail::slot_t].array();
    }
}
} // namespace

TEST_CASE("restoration equality local KKT recovery satisfies full-KKT linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 2);
    slot_value(eq, detail::slot_p) << 0.7, 0.9, 1.2;
    slot_value(eq, detail::slot_n) << 0.8, 1.1, 0.6;
    slot_dual(eq, detail::slot_p) << 1.3, 0.7, 0.9;
    slot_dual(eq, detail::slot_n) << 0.6, 1.2, 1.1;

    vector c(3);
    c << 0.2, -0.4, 0.3;
    vector lambda(3);
    lambda << 0.1, -0.2, 0.15;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;

    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar);

    vector delta_c(3);
    delta_c << -0.11, 0.07, 0.19;
    recover_eq_local_step(delta_c, eq);

    const vector res_c = delta_c - slot_step(eq, detail::slot_p) + slot_step(eq, detail::slot_n) + eq.r_c;
    const vector res_p = -eq.d_multiplier - slot_dual_step(eq, detail::slot_p) + slot_r_stat(eq, detail::slot_p);
    const vector res_n = eq.d_multiplier - slot_dual_step(eq, detail::slot_n) + slot_r_stat(eq, detail::slot_n);
    const vector res_sp = slot_dual(eq, detail::slot_p).cwiseProduct(slot_step(eq, detail::slot_p)) +
                          slot_value(eq, detail::slot_p).cwiseProduct(slot_dual_step(eq, detail::slot_p)) +
                          slot_r_comp(eq, detail::slot_p);
    const vector res_sn = slot_dual(eq, detail::slot_n).cwiseProduct(slot_step(eq, detail::slot_n)) +
                          slot_value(eq, detail::slot_n).cwiseProduct(slot_dual_step(eq, detail::slot_n)) +
                          slot_r_comp(eq, detail::slot_n);
    const auto summary = resto_eq_elastic_constr::linearized_newton_residuals(delta_c, eq);

    REQUIRE(approx_zero(res_c));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
    REQUIRE(summary.inf_prim < 1e-12);
    REQUIRE(summary.inf_stat < 1e-12);
    REQUIRE(summary.inf_comp < 1e-12);
}

TEST_CASE("restoration equality local KKT recovery satisfies affine predictor linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    slot_value(eq, detail::slot_p) << 0.7, 0.9;
    slot_value(eq, detail::slot_n) << 0.8, 1.1;
    slot_dual(eq, detail::slot_p) << 1.3, 0.7;
    slot_dual(eq, detail::slot_n) << 0.6, 1.2;

    vector c(2);
    c << 0.2, -0.4;
    vector lambda(2);
    lambda << 0.1, -0.2;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;

    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    recover_eq_local_step(delta_c, eq);

    const vector res_c = delta_c - slot_step(eq, detail::slot_p) + slot_step(eq, detail::slot_n) + eq.r_c;
    const vector res_p = -eq.d_multiplier - slot_dual_step(eq, detail::slot_p) + slot_r_stat(eq, detail::slot_p);
    const vector res_n = eq.d_multiplier - slot_dual_step(eq, detail::slot_n) + slot_r_stat(eq, detail::slot_n);
    const vector res_sp = slot_dual(eq, detail::slot_p).cwiseProduct(slot_step(eq, detail::slot_p)) +
                          slot_value(eq, detail::slot_p).cwiseProduct(slot_dual_step(eq, detail::slot_p)) +
                          slot_r_comp(eq, detail::slot_p);
    const vector res_sn = slot_dual(eq, detail::slot_n).cwiseProduct(slot_step(eq, detail::slot_n)) +
                          slot_value(eq, detail::slot_n).cwiseProduct(slot_dual_step(eq, detail::slot_n)) +
                          slot_r_comp(eq, detail::slot_n);

    REQUIRE(approx_zero(res_c));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
}

TEST_CASE("restoration equality local KKT recovery satisfies corrector linearization") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    slot_value(eq, detail::slot_p) << 0.7, 0.9;
    slot_value(eq, detail::slot_n) << 0.8, 1.1;
    slot_dual(eq, detail::slot_p) << 1.3, 0.7;
    slot_dual(eq, detail::slot_n) << 0.6, 1.2;

    vector c(2);
    c << 0.2, -0.4;
    vector lambda(2);
    lambda << 0.1, -0.2;
    const scalar_t rho = 2.0;
    const scalar_t mu_bar = 0.3;

    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    recover_eq_local_step(delta_c, eq);

    const vector res_c = delta_c - slot_step(eq, detail::slot_p) + slot_step(eq, detail::slot_n) + eq.r_c;
    const vector res_p = -eq.d_multiplier - slot_dual_step(eq, detail::slot_p) + slot_r_stat(eq, detail::slot_p);
    const vector res_n = eq.d_multiplier - slot_dual_step(eq, detail::slot_n) + slot_r_stat(eq, detail::slot_n);
    const vector res_sp = slot_dual(eq, detail::slot_p).cwiseProduct(slot_step(eq, detail::slot_p)) +
                          slot_value(eq, detail::slot_p).cwiseProduct(slot_dual_step(eq, detail::slot_p)) +
                          slot_r_comp(eq, detail::slot_p);
    const vector res_sn = slot_dual(eq, detail::slot_n).cwiseProduct(slot_step(eq, detail::slot_n)) +
                          slot_value(eq, detail::slot_n).cwiseProduct(slot_dual_step(eq, detail::slot_n)) +
                          slot_r_comp(eq, detail::slot_n);

    REQUIRE(approx_zero(res_c));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
}

TEST_CASE("restoration overlay problem keeps dyn and replaces non-dynamics with standard overlays") {
    auto prob = ocp::create();

    auto [x, y] = sym::states("x", 2);
    auto u = sym::inputs("u", 1);

    prob->add(*x);
    prob->add(*u);
    prob->add(*y);

    auto cost_stage = cost(new generic_cost("stage_cost", approx_order::second));
    dynamic_cast<generic_func &>(*cost_stage).add_argument(u);
    cost_stage->value = [](func_approx_data &d) {
        d.v_(0) += scalar_t(0.5) * d[0].squaredNorm();
    };
    cost_stage->jacobian = [](func_approx_data &d) {
        d.jac_[0].noalias() += d[0].transpose();
    };
    cost_stage->hessian = [](func_approx_data &d) {
        d.lag_hess_[0][0].diagonal().array() += 1.;
    };
    prob->add(*cost_stage);

    auto eq = constr(new generic_constr("eq", approx_order::second, 1));
    eq->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*eq).add_argument(x);
    eq->value = [](func_approx_data &d) { d.v_(0) = d[0](0); };
    eq->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    prob->add(*eq);

    auto iq = constr(new generic_constr("iq", approx_order::second, 1));
    iq->field_hint().is_eq = false;
    dynamic_cast<generic_func &>(*iq).add_argument(u);
    iq->value = [](func_approx_data &d) { d.v_(0) = d[0](0) - scalar_t(1.0); };
    iq->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    prob->add(*iq);

    auto dyn = constr(new generic_constr("dyn", approx_order::first, 2, __dyn));
    dynamic_cast<generic_func &>(*dyn).add_argument(x);
    dynamic_cast<generic_func &>(*dyn).add_argument(u);
    dynamic_cast<generic_func &>(*dyn).add_argument(y);
    dyn->value = [](func_approx_data &d) {
        d.v_ = d[2] - d[0];
    };
    dyn->jacobian = [](func_approx_data &d) {
        d.jac_[0].setZero();
        d.jac_[2].setZero();
        d.jac_[0].diagonal().array() = -1.;
        d.jac_[2].diagonal().array() = 1.;
    };
    prob->add(*dyn);

    prob->wait_until_ready();

    const auto resto = build_restoration_overlay_problem(
        prob,
        restoration_overlay_settings{
            .rho_u = 1e-4,
            .rho_y = 1e-4,
            .rho_eq = 10.0,
            .rho_ineq = 20.0,
        });

    REQUIRE(resto->num(__dyn) == 1);
    REQUIRE(resto->num(__cost) == 1);
    REQUIRE(resto->num(__eq_x) == 0);
    REQUIRE(resto->num(__eq_x_soft) == 1);
    REQUIRE(resto->num(__ineq_xu) == 1);

    const auto *prox = dynamic_cast<const resto_prox_cost *>(resto->exprs(__cost).front().get());
    REQUIRE(prox != nullptr);

    const auto *eq_overlay = dynamic_cast<const resto_eq_elastic_constr *>(resto->exprs(__eq_x_soft).front().get());
    REQUIRE(eq_overlay != nullptr);
    REQUIRE(eq_overlay->source()->name() == eq->name());

    const auto *ineq_overlay = dynamic_cast<const resto_ineq_elastic_ipm_constr *>(resto->exprs(__ineq_xu).front().get());
    REQUIRE(ineq_overlay != nullptr);
    REQUIRE(ineq_overlay->source()->name() == iq->name());
}

TEST_CASE("restoration soft-equality overlays are built and keep synced multipliers through initialization") {
    auto prob = ocp::create();

    auto x = sym::state("x_resto_soft", 1);
    prob->add(*x);

    auto soft_eq = constr(new generic_constr("soft_eq", approx_order::second, 1));
    soft_eq->field_hint().is_eq = true;
    soft_eq->field_hint().is_soft = true;
    dynamic_cast<generic_func &>(*soft_eq).add_argument(x);
    soft_eq->value = [](func_approx_data &d) { d.v_(0) = d[0](0) - scalar_t(0.25); };
    soft_eq->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    prob->add(*soft_eq);

    prob->wait_until_ready();

    const auto resto_prob = build_restoration_overlay_problem(
        prob,
        restoration_overlay_settings{
            .rho_u = 1e-4,
            .rho_y = 1e-4,
            .rho_eq = 10.0,
            .rho_ineq = 20.0,
        });

    REQUIRE(resto_prob->num(__eq_x_soft) == 1);
    const auto *eq_overlay = dynamic_cast<const resto_eq_elastic_constr *>(resto_prob->exprs(__eq_x_soft).front().get());
    REQUIRE(eq_overlay != nullptr);
    REQUIRE(eq_overlay->source()->name() == soft_eq->name());
    REQUIRE(eq_overlay->source()->field() == __eq_x_soft);

    ns_sqp::data outer(prob);
    ns_sqp::data resto(resto_prob);
    ns_sqp::settings_t ws;

    resto.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
        c.setup_workspace_data(fd, &ws);
    });

    REQUIRE(outer.dense().dual_[__eq_x_soft].size() == 1);
    outer.dense().dual_[__eq_x_soft](0) = scalar_t(2.5);

    sync_outer_to_restoration_state(outer, resto, scalar_t(1.0), &ws.mu);

    bool saw_overlay = false;
    resto.for_each(__eq_x_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        saw_overlay = true;
        REQUIRE(overlay.source()->name() == soft_eq->name());
        REQUIRE(approx_scalar(d.multiplier_(0), scalar_t(2.5)));
    });
    REQUIRE(saw_overlay);

    solver::ineq_soft::bind_runtime(&resto);
    resto.update_approximation(node_data::update_mode::eval_val, true);

    saw_overlay = false;
    resto.for_each(__eq_x_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        saw_overlay = true;
        REQUIRE(overlay.source()->name() == soft_eq->name());
        REQUIRE(approx_scalar(d.multiplier_(0), scalar_t(0.0)));
        REQUIRE(approx_scalar(d.multiplier_backup(0), scalar_t(0.0)));
    });
    REQUIRE(saw_overlay);
}

TEST_CASE("box helper stores only base box dimension in ocp") {
    auto [x, _y] = sym::states("x_box", 2);
    const vector lb = (vector(2) << -1.0, 0.5).finished();
    const vector ub = (vector(2) << 2.0, 3.0).finished();

    auto box = ineq_constr::create("x_box_bound", var_inarg_list{*x}, static_cast<const cs::SX &>(x), lb, ub);
    REQUIRE(box->dim() == 2);
    REQUIRE(box->field_hint().is_eq == utils::optional_bool::False);

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);
}

TEST_CASE("box helper keeps generic casadi lowering for single-arg linear selection") {
    auto [x, _y] = sym::states("x_sel", 5);
    const vector lb = (vector(2) << -1.0, 2.0).finished();
    const vector ub = (vector(2) << 4.0, 5.0).finished();

    auto bound = ineq_constr::create(
        "x_sel_bound",
        var_inarg_list{*x},
        cs::SX::vertcat(std::vector<cs::SX>{x(0), x(3)}),
        lb,
        ub);
    REQUIRE(bound->dim() == 2);
    REQUIRE(bound->field_hint().is_eq == utils::optional_bool::False);
    REQUIRE(bound->get_codegen_task() != nullptr);

    REQUIRE(bound->finalize());
    REQUIRE(bound->field() == __ineq_x);
}

TEST_CASE("box helper drops unbounded sides row-wise") {
    auto [x, _y] = sym::states("x_half_box", 2);
    const vector lb = (vector(2) << -std::numeric_limits<scalar_t>::infinity(), 0.5).finished();
    const vector ub = (vector(2) << 2.0, std::numeric_limits<scalar_t>::infinity()).finished();

    auto box = ineq_constr::create("x_half_box_bound", var_inarg_list{*x}, static_cast<const cs::SX &>(x), lb, ub);
    REQUIRE(box->dim() == 2);
    REQUIRE(box->field_hint().is_eq == utils::optional_bool::False);

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);
}

TEST_CASE("box helper accepts symbolic parameter bounds through generic casadi lowering") {
    auto [x, _y] = sym::states("x_param_box", 2);
    auto p = sym::params("p_box", 2);

    auto box = ineq_constr::create(
        "x_param_box_bound",
        var_inarg_list{*x, *p},
        static_cast<const cs::SX &>(x),
        -std::numeric_limits<scalar_t>::infinity(),
        static_cast<const cs::SX &>(p));
    REQUIRE(box->dim() == 2);
    REQUIRE(box->field_hint().is_eq == utils::optional_bool::False);
    REQUIRE(box->get_codegen_task() != nullptr);

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);
}

TEST_CASE("ipm box preserves native metadata for parameter bounds") {
    auto [x, _y] = sym::states("x_native_box", 2);
    auto p = sym::params("p_native_box", 2);

    constr box(ineq_constr::create(
        "x_native_box_bound",
        var_inarg_list{*x, *p},
        static_cast<const cs::SX &>(x),
        -std::numeric_limits<scalar_t>::infinity(),
        static_cast<const cs::SX &>(p)));

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);

    const auto *ipm_box = dynamic_cast<const solver::ipm_constr *>(box.get());
    REQUIRE(ipm_box != nullptr);
    REQUIRE(ipm_box->box_info() != nullptr);
    REQUIRE(ipm_box->box_info()->base_dim == static_cast<size_t>(box->dim()));
    REQUIRE(ipm_box->box_info()->base_dim == 2);
}

TEST_CASE("ordinary ipm inequality is normalized into an upper half-box") {
    auto prob = ocp::create();
    auto [x, _y] = sym::states("x_half_box_ipm", 1);
    auto u = sym::inputs("u_half_box_ipm", 1);

    prob->add(*x);
    prob->add(*u);

    auto iq = ineq_constr::create("iq_half_box_ipm", approx_order::first, 1);
    dynamic_cast<generic_func &>(*iq).add_argument(x);
    dynamic_cast<generic_func &>(*iq).add_argument(u);
    iq->value = [](func_approx_data &d) { d.v_(0) = d[0](0) - d[1](0) - scalar_t(0.25); };
    iq->jacobian = [](func_approx_data &d) {
        d.jac_[0](0, 0) = 1.;
        d.jac_[1](0, 0) = -1.;
    };
    prob->add(*iq);
    prob->wait_until_ready();

    const auto *ipm_iq = dynamic_cast<const solver::ipm_constr *>(iq.get());
    REQUIRE(ipm_iq != nullptr);
    REQUIRE(ipm_iq->box_info() != nullptr);
    REQUIRE(ipm_iq->box_info()->base_dim == 1);
    REQUIRE(ipm_iq->box_info()->present_mask[box_side::ub].size() == 1);
    REQUIRE(ipm_iq->box_info()->present_mask[box_side::lb].size() == 1);
    REQUIRE(ipm_iq->box_info()->present_mask[box_side::ub](0));
    REQUIRE_FALSE(ipm_iq->box_info()->present_mask[box_side::lb](0));

    ns_sqp::data stage(prob);
    ns_sqp::settings_t ws;
    stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
        c.setup_workspace_data(fd, &ws);
    });

    stage.sym_val().value_[__x](0) = 1.0;
    stage.sym_val().value_[__u](0) = 0.5;

    solver::ineq_soft::bind_runtime(&stage);
    stage.update_approximation(node_data::update_mode::eval_all, true);

    bool saw_ineq = false;
    stage.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d) {
        saw_ineq = true;
        REQUIRE(d.boxed());
        REQUIRE(approx_scalar(d.box_side_[box_side::ub]->residual(0), 0.25));
        REQUIRE(approx_scalar(d.jac_[0](0, 0), 1.0));
        REQUIRE(approx_scalar(d.jac_[1](0, 0), -1.0));
    });
    REQUIRE(saw_ineq);
}

TEST_CASE("native ipm box groups side workspace in approx data") {
    auto prob = ocp::create();
    auto [x, _y] = sym::states("x_runtime_native_box", 2);

    prob->add(*x);

    constr box(ineq_constr::create(
        "x_runtime_native_box_bound",
        var_inarg_list{*x},
        static_cast<const cs::SX &>(x),
        (vector(2) << -1.0, -2.0).finished(),
        (vector(2) << 3.0, 4.0).finished()));
    prob->add(*box);
    prob->wait_until_ready();

    ns_sqp::data stage(prob);
    ns_sqp::settings_t ws;
    stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
        c.setup_workspace_data(fd, &ws);
    });

    solver::ineq_soft::bind_runtime(&stage);
    stage.update_approximation(node_data::update_mode::eval_all, true);

    bool saw_box = false;
    stage.for_each(__ineq_x, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d) {
        saw_box = true;
        REQUIRE(d.boxed());
        REQUIRE(d.box_side_[box_side::ub] != nullptr);
        REQUIRE(d.box_side_[box_side::lb] != nullptr);
        REQUIRE(d.box_side_[box_side::ub]->slack.size() == 2);
        REQUIRE(d.box_side_[box_side::lb]->slack.size() == 2);
        auto &upper = dynamic_cast<solver::ipm_constr::approx_data::side_data &>(*d.box_side_[box_side::ub]);
        auto &lower = dynamic_cast<solver::ipm_constr::approx_data::side_data &>(*d.box_side_[box_side::lb]);
        REQUIRE(upper.diag_scaling.size() == 2);
        REQUIRE(lower.diag_scaling.size() == 2);
        REQUIRE(upper.scaled_res.size() == 2);
        REQUIRE(lower.scaled_res.size() == 2);
        REQUIRE_FALSE(upper.scaled_res.hasNaN());
        REQUIRE_FALSE(lower.scaled_res.hasNaN());
    });
    REQUIRE(saw_box);
}

TEST_CASE("native ipm box backup and restore use side workspace") {
    auto prob = ocp::create();
    auto [x, _y] = sym::states("x_runtime_native_box_restore", 2);

    prob->add(*x);

    constr box(ineq_constr::create(
        "x_runtime_native_box_restore_bound",
        var_inarg_list{*x},
        static_cast<const cs::SX &>(x),
        (vector(2) << -1.0, -2.0).finished(),
        (vector(2) << 3.0, 4.0).finished()));
    prob->add(*box);
    prob->wait_until_ready();

    ns_sqp::data stage(prob);
    ns_sqp::settings_t ws;
    stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
        c.setup_workspace_data(fd, &ws);
    });

    solver::ineq_soft::bind_runtime(&stage);
    stage.update_approximation(node_data::update_mode::eval_all, true);

    bool saw_box = false;
    stage.for_each(__ineq_x, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        saw_box = true;
        auto &upper = *d.box_side_[box_side::ub];
        auto &lower = *d.box_side_[box_side::lb];

        upper.slack << 1.0, 2.0;
        lower.slack << 3.0, 4.0;
        upper.multiplier << 0.5, 0.6;
        lower.multiplier << 0.7, 0.8;
        ipm.backup_trial_state(d);

        const vector upper_slack_saved = upper.slack;
        const vector lower_slack_saved = lower.slack;
        const vector upper_mult_saved = upper.multiplier;
        const vector lower_mult_saved = lower.multiplier;

        upper.slack.setConstant(-1.0);
        lower.slack.setConstant(-1.0);
        upper.multiplier.setConstant(-2.0);
        lower.multiplier.setConstant(-2.0);

        ipm.restore_trial_state(d);

        REQUIRE(approx_zero(upper.slack - upper_slack_saved));
        REQUIRE(approx_zero(lower.slack - lower_slack_saved));
        REQUIRE(approx_zero(upper.multiplier - upper_mult_saved));
        REQUIRE(approx_zero(lower.multiplier - lower_mult_saved));
        REQUIRE(upper.slack.size() == 2);
        REQUIRE(lower.slack.size() == 2);
        REQUIRE(upper.multiplier.size() == 2);
        REQUIRE(lower.multiplier.size() == 2);
    });
    REQUIRE(saw_box);
}

TEST_CASE("boxed outer ipm runtime matches equivalent stacked one-sided rows") {
    auto build_box_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_box", 2);
        prob->add(*u);
        const vector lb = (vector(2) << -1.0, -2.0).finished();
        const vector ub = (vector(2) << 3.0, 4.0).finished();
        prob->add(*ineq_constr::create(
            "u_outer_box_bound",
            var_inarg_list{*u},
            static_cast<const cs::SX &>(u),
            lb,
            ub));
        prob->wait_until_ready();
        return prob;
    };
    auto build_stacked_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_stacked", 2);
        prob->add(*u);
        const cs::SX g = cs::SX::vertcat(std::vector<cs::SX>{
            u(0) - scalar_t(3.0),
            scalar_t(-1.0) - u(0),
            u(1) - scalar_t(4.0),
            scalar_t(-2.0) - u(1),
        });
        prob->add(*ineq_constr::create("u_outer_stacked_bound", var_inarg_list{*u}, g));
        prob->wait_until_ready();
        return prob;
    };

    const auto box_prob = build_box_prob();
    const auto stacked_prob = build_stacked_prob();

    ns_sqp::data outer_box(box_prob);
    ns_sqp::data outer_stacked(stacked_prob);
    ns_sqp::settings_t ws;
    ws.mu = scalar_t(0.3);
    ws.alpha_primal = scalar_t(0.4);
    ws.alpha_dual = scalar_t(0.35);

    auto setup_stage = [&](ns_sqp::data &stage) {
        stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
            c.setup_workspace_data(fd, &ws);
        });
        solver::ineq_soft::bind_runtime(&stage);
    };
    setup_stage(outer_box);
    setup_stage(outer_stacked);

    outer_box.sym_val().value_[__u] << 0.25, -0.75;
    outer_stacked.sym_val().value_[__u] = outer_box.sym_val().value_[__u];

    outer_box.update_approximation(node_data::update_mode::eval_all, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_all, true);

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });

    outer_box.trial_prim_step[__u] << 0.2, -0.15;
    outer_stacked.trial_prim_step[__u] = outer_box.trial_prim_step[__u];

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.finalize_newton_step(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.finalize_newton_step(d);
    });

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &box_ipm, solver::ipm_constr::approx_data &d) {
        const auto &ub = *d.box_side_[box_side::ub];
        const auto &lb = *d.box_side_[box_side::lb];
        outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &stacked_ipm, solver::ipm_constr::approx_data &s) {
            const auto &side = *s.box_side_[box_side::ub];
            REQUIRE((ub.slack - gather_entries(side.slack, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.slack - gather_entries(side.slack, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((ub.multiplier - gather_entries(side.multiplier, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.multiplier - gather_entries(side.multiplier, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((ub.d_slack - gather_entries(side.d_slack, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.d_slack - gather_entries(side.d_slack, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((ub.d_multiplier - gather_entries(side.d_multiplier, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.d_multiplier - gather_entries(side.d_multiplier, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE(approx_scalar(box_ipm.search_penalty(d), stacked_ipm.search_penalty(s), 1e-12));
            REQUIRE(approx_scalar(box_ipm.search_penalty_dir_deriv(d), stacked_ipm.search_penalty_dir_deriv(s), 1e-12));
            vector stacked_view(2);
            stacked_view << std::max(s.v_(0), s.v_(1)), std::max(s.v_(2), s.v_(3));
            REQUIRE((d.v_ - stacked_view).cwiseAbs().maxCoeff() < 1e-12);
        });
    });

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.apply_affine_step(d, &ws);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.apply_affine_step(d, &ws);
    });

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &box_ipm, solver::ipm_constr::approx_data &d) {
        const auto &ub = *d.box_side_[box_side::ub];
        const auto &lb = *d.box_side_[box_side::lb];
        outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &stacked_ipm, solver::ipm_constr::approx_data &s) {
            const auto &side = *s.box_side_[box_side::ub];
            REQUIRE((ub.slack - gather_entries(side.slack, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.slack - gather_entries(side.slack, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((ub.multiplier - gather_entries(side.multiplier, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.multiplier - gather_entries(side.multiplier, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE(approx_scalar(box_ipm.search_penalty(d), stacked_ipm.search_penalty(s), 1e-12));
            vector stacked_view(2);
            stacked_view << std::max(s.v_(0), s.v_(1)), std::max(s.v_(2), s.v_(3));
            REQUIRE((d.v_ - stacked_view).cwiseAbs().maxCoeff() < 1e-12);
        });
    });
}

TEST_CASE("boxed outer ipm derivative propagation matches equivalent stacked one-sided rows") {
    auto build_box_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_deriv_box", 2);
        prob->add(*u);
        const vector lb = (vector(2) << -1.0, -2.0).finished();
        const vector ub = (vector(2) << 3.0, 4.0).finished();
        prob->add(*ineq_constr::create(
            "u_outer_deriv_box_bound",
            var_inarg_list{*u},
            static_cast<const cs::SX &>(u),
            lb,
            ub));
        prob->wait_until_ready();
        return prob;
    };
    auto build_stacked_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_deriv_stacked", 2);
        prob->add(*u);
        const cs::SX g = cs::SX::vertcat(std::vector<cs::SX>{
            u(0) - scalar_t(3.0),
            scalar_t(-1.0) - u(0),
            u(1) - scalar_t(4.0),
            scalar_t(-2.0) - u(1),
        });
        prob->add(*ineq_constr::create("u_outer_deriv_stacked_bound", var_inarg_list{*u}, g));
        prob->wait_until_ready();
        return prob;
    };

    const auto box_prob = build_box_prob();
    const auto stacked_prob = build_stacked_prob();

    ns_sqp::data outer_box(box_prob);
    ns_sqp::data outer_stacked(stacked_prob);
    ns_sqp::settings_t ws;
    ws.mu = scalar_t(0.3);

    auto setup_stage = [&](ns_sqp::data &stage) {
        stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
            c.setup_workspace_data(fd, &ws);
        });
        solver::ineq_soft::bind_runtime(&stage);
    };
    setup_stage(outer_box);
    setup_stage(outer_stacked);

    outer_box.sym_val().value_[__u] << 0.25, -0.75;
    outer_stacked.sym_val().value_[__u] = outer_box.sym_val().value_[__u];

    outer_box.update_approximation(node_data::update_mode::eval_all, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_all, true);

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });

    for (auto field : primal_fields) {
        outer_box.dense().lag_jac_corr_[field].setZero();
        outer_stacked.dense().lag_jac_corr_[field].setZero();
        for (auto inner : primal_fields) {
            outer_box.dense().hessian_modification_[field][inner].setZero();
            outer_stacked.dense().hessian_modification_[field][inner].setZero();
        }
    }

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.jacobian_impl(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.jacobian_impl(d);
    });

    REQUIRE((outer_box.dense().lag_jac_corr_[__u] - outer_stacked.dense().lag_jac_corr_[__u]).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((outer_box.dense().hessian_modification_[__u][__u].dense() -
             outer_stacked.dense().hessian_modification_[__u][__u].dense())
                .cwiseAbs()
                .maxCoeff() < 1e-12);
}

TEST_CASE("boxed outer ipm predictor-corrector matches equivalent stacked one-sided rows") {
    auto build_box_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_pc_box", 2);
        prob->add(*u);
        const vector lb = (vector(2) << -1.0, -2.0).finished();
        const vector ub = (vector(2) << 3.0, 4.0).finished();
        prob->add(*ineq_constr::create(
            "u_outer_pc_box_bound",
            var_inarg_list{*u},
            static_cast<const cs::SX &>(u),
            lb,
            ub));
        prob->wait_until_ready();
        return prob;
    };
    auto build_stacked_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_outer_pc_stacked", 2);
        prob->add(*u);
        const cs::SX g = cs::SX::vertcat(std::vector<cs::SX>{
            u(0) - scalar_t(3.0),
            scalar_t(-1.0) - u(0),
            u(1) - scalar_t(4.0),
            scalar_t(-2.0) - u(1),
        });
        prob->add(*ineq_constr::create("u_outer_pc_stacked_bound", var_inarg_list{*u}, g));
        prob->wait_until_ready();
        return prob;
    };

    const auto box_prob = build_box_prob();
    const auto stacked_prob = build_stacked_prob();

    ns_sqp::data outer_box(box_prob);
    ns_sqp::data outer_stacked(stacked_prob);
    ns_sqp::settings_t ws_box;
    ns_sqp::settings_t ws_stacked;
    ws_box.ipm.mu = scalar_t(0.3);
    ws_stacked.ipm.mu = scalar_t(0.3);
    ws_box.ipm.mu_method = solver::ipm_config::mehrotra_predictor_corrector;
    ws_stacked.ipm.mu_method = solver::ipm_config::mehrotra_predictor_corrector;
    ws_box.alpha_primal = scalar_t(0.4);
    ws_box.alpha_dual = scalar_t(0.35);
    ws_stacked.alpha_primal = ws_box.alpha_primal;
    ws_stacked.alpha_dual = ws_box.alpha_dual;

    auto setup_stage = [&](ns_sqp::data &stage, ns_sqp::settings_t &ws) {
        stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
            c.setup_workspace_data(fd, &ws);
        });
        solver::ineq_soft::bind_runtime(&stage);
    };
    setup_stage(outer_box, ws_box);
    setup_stage(outer_stacked, ws_stacked);

    outer_box.sym_val().value_[__u] << 0.25, -0.75;
    outer_stacked.sym_val().value_[__u] = outer_box.sym_val().value_[__u];

    outer_box.update_approximation(node_data::update_mode::eval_all, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_all, true);

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.initialize(d);
    });

    outer_box.trial_prim_step[__u] << 0.2, -0.15;
    outer_stacked.trial_prim_step[__u] = outer_box.trial_prim_step[__u];

    ns_sqp::settings_t::worker worker_box;
    ns_sqp::settings_t::worker worker_stacked;
    worker_box.as<solver::linesearch_config>().copy_from(ws_box);
    worker_stacked.as<solver::linesearch_config>().copy_from(ws_stacked);

    ws_box.ipm_start_predictor_computation();
    ws_stacked.ipm_start_predictor_computation();
    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.finalize_newton_step(d);
        ipm.finalize_predictor_step(d, &worker_box);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.finalize_newton_step(d);
        ipm.finalize_predictor_step(d, &worker_stacked);
    });
    ws_box.ipm_end_predictor_computation();
    ws_stacked.ipm_end_predictor_computation();

    const auto &worker_box_ipm = worker_box.as<solver::ipm_config::worker_type>();
    const auto &worker_stacked_ipm = worker_stacked.as<solver::ipm_config::worker_type>();
    REQUIRE(worker_box_ipm.n_ipm_cstr == worker_stacked_ipm.n_ipm_cstr);
    REQUIRE(approx_scalar(worker_box_ipm.prev_aff_comp, worker_stacked_ipm.prev_aff_comp, 1e-12));
    REQUIRE(approx_scalar(worker_box_ipm.post_aff_comp, worker_stacked_ipm.post_aff_comp, 1e-12));

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d) {
        const auto &ub = static_cast<const solver::ipm_constr::approx_data::side_data &>(*d.box_side_[box_side::ub]);
        const auto &lb = static_cast<const solver::ipm_constr::approx_data::side_data &>(*d.box_side_[box_side::lb]);
        outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &s) {
            const auto &side = static_cast<const solver::ipm_constr::approx_data::side_data &>(*s.box_side_[box_side::ub]);
            REQUIRE((ub.corrector - gather_entries(side.corrector, {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.corrector - gather_entries(side.corrector, {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
        });
    });

    outer_box.dense().lag_jac_corr_[__u].setZero();
    outer_stacked.dense().lag_jac_corr_[__u].setZero();
    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.apply_corrector_step(d);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &ipm, solver::ipm_constr::approx_data &d) {
        ipm.apply_corrector_step(d);
    });

    REQUIRE((outer_box.dense().lag_jac_corr_[__u] - outer_stacked.dense().lag_jac_corr_[__u]).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("box helper rejects bounds that depend on primal in_args") {
    auto [x, _y] = sym::states("x_non_native_box", 2);
    auto p = sym::params("p_non_native_box", 2);

    REQUIRE_THROWS_WITH(
        ineq_constr::create(
            "x_non_native_box_bound",
            var_inarg_list{*x, *p},
            static_cast<const cs::SX &>(x),
            static_cast<const cs::SX &>(p) - static_cast<const cs::SX &>(x),
            static_cast<const cs::SX &>(p)),
        Catch::Matchers::ContainsSubstring("must not depend on primal in_args"));
}

TEST_CASE("box helper rejects symbolic bounds that are not direct non-primal in_args") {
    auto [x, _y] = sym::states("x_non_direct_box", 2);
    auto p = sym::params("p_non_direct_box", 2);

    REQUIRE_THROWS_WITH(
        ineq_constr::create(
            "x_non_direct_box_bound",
            var_inarg_list{*x, *p},
            static_cast<const cs::SX &>(x),
            static_cast<const cs::SX &>(p) + 1.0,
            static_cast<const cs::SX &>(p)),
        Catch::Matchers::ContainsSubstring("direct non-primal in_arg"));
}

TEST_CASE("box helper rejects symbolic bounds whose dependencies are missing from in_args") {
    auto [x, _y] = sym::states("x_missing_box", 2);
    auto p = sym::params("p_missing_box", 2);

    REQUIRE_THROWS_WITH(
        ineq_constr::create(
            "x_missing_box_bound",
            var_inarg_list{*x},
            static_cast<const cs::SX &>(x),
            -std::numeric_limits<scalar_t>::infinity(),
            static_cast<const cs::SX &>(p)),
        Catch::Matchers::ContainsSubstring("not listed in in_args"));
}

TEST_CASE("restoration overlay wraps box-lowered ipm inequality without special casing") {
    auto prob = ocp::create();

    auto [x, y] = sym::states("x", 2);
    auto u = sym::inputs("u", 1);

    prob->add(*x);
    prob->add(*u);
    prob->add(*y);

    auto dyn = constr(new generic_constr("dyn", approx_order::first, 2, __dyn));
    dynamic_cast<generic_func &>(*dyn).add_argument(x);
    dynamic_cast<generic_func &>(*dyn).add_argument(u);
    dynamic_cast<generic_func &>(*dyn).add_argument(y);
    dyn->value = [](func_approx_data &d) { d.v_ = d[2] - d[0]; };
    dyn->jacobian = [](func_approx_data &d) {
        d.jac_[0].setZero();
        d.jac_[2].setZero();
        d.jac_[0].diagonal().array() = -1.;
        d.jac_[2].diagonal().array() = 1.;
    };
    prob->add(*dyn);

    const vector lb = (vector(2) << -1.0, -2.0).finished();
    const vector ub = (vector(2) << 3.0, 4.0).finished();
    constr box(ineq_constr::create(
        "x_box_bound",
        var_inarg_list{*u},
        cs::SX::vertcat(std::vector<cs::SX>{u(0), scalar_t(2.0) * u(0)}),
        lb,
        ub));
    prob->add(*box);

    prob->wait_until_ready();

    const auto resto = build_restoration_overlay_problem(
        prob,
        restoration_overlay_settings{
            .rho_u = 1e-4,
            .rho_y = 1e-4,
            .rho_eq = 10.0,
            .rho_ineq = 20.0,
        });

    REQUIRE(resto->num(__ineq_xu) == 1);
    const auto *ineq_overlay = dynamic_cast<const resto_ineq_elastic_ipm_constr *>(resto->exprs(__ineq_xu).front().get());
    REQUIRE(ineq_overlay != nullptr);
    REQUIRE(ineq_overlay->source()->name() == box->name());
    REQUIRE(ineq_overlay->source()->dim() == 2);
}
TEST_CASE("restoration inequality local KKT recovery satisfies reduced linearization") {
    detail::ineq_local_state iq;
    resize_ineq_state(iq, 2, 1);
    slot_value(iq, detail::slot_t) << 1.1, 0.8, 1.4;
    slot_value(iq, detail::slot_p) << 0.8, 1.0, 1.3;
    slot_value(iq, detail::slot_n) << 0.9, 0.7, 1.2;
    slot_dual(iq, detail::slot_t) << 0.4, 0.6, 0.5;
    slot_dual(iq, detail::slot_p) << 0.9, 0.8, 0.7;
    slot_dual(iq, detail::slot_n) << 0.6, 0.9, 0.5;

    vector g(3);
    g << -0.2, 0.5, -0.1;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;
    iq.base_residual = g;
    iq.side[box_side::ub].residual = g;

    resto_ineq_elastic_ipm_constr::compute_local_model(iq, make_half_box_spec(g.size()), rho, mu_bar);

    vector delta_g(3);
    delta_g << 0.08, -0.12, 0.04;
    recover_ineq_local_step(delta_g, iq);

    const vector res_d = delta_g + slot_step(iq, detail::slot_t) - slot_step(iq, detail::slot_p) +
                         slot_step(iq, detail::slot_n) + iq.side[box_side::ub].r_d;
    const vector res_p =
        -slot_dual_step(iq, detail::slot_t) - slot_dual_step(iq, detail::slot_p) + slot_r_stat(iq, detail::slot_p);
    const vector res_n =
        slot_dual_step(iq, detail::slot_t) - slot_dual_step(iq, detail::slot_n) + slot_r_stat(iq, detail::slot_n);
    const vector res_st = slot_dual(iq, detail::slot_t).cwiseProduct(slot_step(iq, detail::slot_t)) +
                          slot_value(iq, detail::slot_t).cwiseProduct(slot_dual_step(iq, detail::slot_t)) +
                          slot_r_comp(iq, detail::slot_t);
    const vector res_sp = slot_dual(iq, detail::slot_p).cwiseProduct(slot_step(iq, detail::slot_p)) +
                          slot_value(iq, detail::slot_p).cwiseProduct(slot_dual_step(iq, detail::slot_p)) +
                          slot_r_comp(iq, detail::slot_p);
    const vector res_sn = slot_dual(iq, detail::slot_n).cwiseProduct(slot_step(iq, detail::slot_n)) +
                          slot_value(iq, detail::slot_n).cwiseProduct(slot_dual_step(iq, detail::slot_n)) +
                          slot_r_comp(iq, detail::slot_n);
    const auto summary = resto_ineq_elastic_ipm_constr::linearized_newton_residuals(delta_g, iq);

    REQUIRE(approx_zero(res_d));
    REQUIRE(approx_zero(res_p));
    REQUIRE(approx_zero(res_n));
    REQUIRE(approx_zero(res_st));
    REQUIRE(approx_zero(res_sp));
    REQUIRE(approx_zero(res_sn));
    REQUIRE(summary.inf_prim < 1e-12);
    REQUIRE(summary.inf_stat < 1e-12);
    REQUIRE(summary.inf_comp < 1e-12);
}

TEST_CASE("restoration boxed local model matches equivalent stacked one-sided rows") {
    detail::ineq_local_state boxed;
    resize_ineq_state(boxed, 2, 0);
    boxed.present_mask[box_side::ub] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(2, true);
    boxed.present_mask[box_side::lb] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(2, true);
    boxed.side[box_side::ub].residual << -0.4, 0.2;
    boxed.side[box_side::lb].residual << -0.1, -0.6;
    slot_value(boxed, detail::slot_t, box_side::ub) << 1.2, 1.4;
    slot_value(boxed, detail::slot_p, box_side::ub) << 0.7, 0.8;
    slot_value(boxed, detail::slot_n, box_side::ub) << 0.9, 1.1;
    slot_dual(boxed, detail::slot_t, box_side::ub) << 0.6, 0.5;
    slot_dual(boxed, detail::slot_p, box_side::ub) << 1.0, 0.9;
    slot_dual(boxed, detail::slot_n, box_side::ub) << 0.8, 0.7;
    slot_value(boxed, detail::slot_t, box_side::lb) << 1.3, 1.5;
    slot_value(boxed, detail::slot_p, box_side::lb) << 0.6, 0.9;
    slot_value(boxed, detail::slot_n, box_side::lb) << 1.0, 1.2;
    slot_dual(boxed, detail::slot_t, box_side::lb) << 0.55, 0.65;
    slot_dual(boxed, detail::slot_p, box_side::lb) << 0.95, 0.85;
    slot_dual(boxed, detail::slot_n, box_side::lb) << 0.75, 0.8;

    detail::ineq_local_state stacked;
    resize_ineq_state(stacked, 4, 0);
    stacked.present_mask[box_side::ub] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(4, true);
    stacked.present_mask[box_side::lb] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(4, false);
    stacked.side[box_side::ub].residual << boxed.side[box_side::ub].residual, boxed.side[box_side::lb].residual;
    slot_value(stacked, detail::slot_t) << slot_value(boxed, detail::slot_t, box_side::ub),
        slot_value(boxed, detail::slot_t, box_side::lb);
    slot_value(stacked, detail::slot_p) << slot_value(boxed, detail::slot_p, box_side::ub),
        slot_value(boxed, detail::slot_p, box_side::lb);
    slot_value(stacked, detail::slot_n) << slot_value(boxed, detail::slot_n, box_side::ub),
        slot_value(boxed, detail::slot_n, box_side::lb);
    slot_dual(stacked, detail::slot_t) << slot_dual(boxed, detail::slot_t, box_side::ub),
        slot_dual(boxed, detail::slot_t, box_side::lb);
    slot_dual(stacked, detail::slot_p) << slot_dual(boxed, detail::slot_p, box_side::ub),
        slot_dual(boxed, detail::slot_p, box_side::lb);
    slot_dual(stacked, detail::slot_n) << slot_dual(boxed, detail::slot_n, box_side::ub),
        slot_dual(boxed, detail::slot_n, box_side::lb);

    ineq_constr::box_spec full_box;
    full_box.base_dim = 2;
    full_box.present_mask[box_side::ub] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(2, true);
    full_box.present_mask[box_side::lb] = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(2, true);
    full_box.has_side[box_side::ub] = true;
    full_box.has_side[box_side::lb] = true;

    const scalar_t rho = 4.0;
    const scalar_t mu_bar = 0.3;
    resto_ineq_elastic_ipm_constr::compute_local_model(boxed, full_box, rho, mu_bar);
    resto_ineq_elastic_ipm_constr::compute_local_model(stacked, make_half_box_spec(4), rho, mu_bar);

    vector schur_rhs_net_expected(2);
    schur_rhs_net_expected <<
        stacked.side[box_side::ub].schur_rhs(0) - stacked.side[box_side::ub].schur_rhs(2),
        stacked.side[box_side::ub].schur_rhs(1) - stacked.side[box_side::ub].schur_rhs(3);
    vector schur_inv_sum_expected(2);
    schur_inv_sum_expected <<
        stacked.side[box_side::ub].schur_inv_diag(0) + stacked.side[box_side::ub].schur_inv_diag(2),
        stacked.side[box_side::ub].schur_inv_diag(1) + stacked.side[box_side::ub].schur_inv_diag(3);
    vector primal_view_expected(2);
    primal_view_expected <<
        std::max(boxed.side[box_side::ub].r_d(0), boxed.side[box_side::lb].r_d(0)),
        std::max(boxed.side[box_side::ub].r_d(1), boxed.side[box_side::lb].r_d(1));

    REQUIRE((boxed.primal_view - primal_view_expected).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((boxed.schur_rhs_net - schur_rhs_net_expected).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((boxed.schur_inv_diag_sum - schur_inv_sum_expected).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("restoration boxed overlay syncs back to outer ipm like equivalent stacked rows") {
    auto build_box_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_sync_box", 2);
        prob->add(*u);
        const vector lb = (vector(2) << -1.0, -2.0).finished();
        const vector ub = (vector(2) << 3.0, 4.0).finished();
        prob->add(*ineq_constr::create(
            "u_box",
            var_inarg_list{*u},
            static_cast<const cs::SX &>(u),
            lb,
            ub));
        prob->wait_until_ready();
        return prob;
    };
    auto build_stacked_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_sync_stacked", 2);
        prob->add(*u);
        const cs::SX g = cs::SX::vertcat(std::vector<cs::SX>{
            u(0) - scalar_t(3.0),
            scalar_t(-1.0) - u(0),
            u(1) - scalar_t(4.0),
            scalar_t(-2.0) - u(1),
        });
        prob->add(*ineq_constr::create("u_stacked", var_inarg_list{*u}, g));
        prob->wait_until_ready();
        return prob;
    };

    const auto box_prob = build_box_prob();
    const auto stacked_prob = build_stacked_prob();
    const auto resto_box_prob = build_restoration_overlay_problem(box_prob, restoration_overlay_settings{.rho_ineq = 20.0});
    const auto resto_stacked_prob = build_restoration_overlay_problem(stacked_prob, restoration_overlay_settings{.rho_ineq = 20.0});

    ns_sqp::data outer_box(box_prob);
    ns_sqp::data resto_box(resto_box_prob);
    ns_sqp::data outer_stacked(stacked_prob);
    ns_sqp::data resto_stacked(resto_stacked_prob);
    ns_sqp::settings_t ws;
    ws.mu = scalar_t(0.3);

    auto setup_stage = [&](ns_sqp::data &stage) {
        stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
            c.setup_workspace_data(fd, &ws);
        });
        solver::ineq_soft::bind_runtime(&stage);
    };
    setup_stage(outer_box);
    setup_stage(resto_box);
    setup_stage(outer_stacked);
    setup_stage(resto_stacked);

    outer_box.sym_val().value_[__u] << 0.25, -0.75;
    outer_stacked.sym_val().value_[__u] = outer_box.sym_val().value_[__u];
    outer_box.update_approximation(node_data::update_mode::eval_all, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_all, true);

    sync_outer_to_restoration_state(outer_box, resto_box, scalar_t(1.0), &ws.mu);
    sync_outer_to_restoration_state(outer_stacked, resto_stacked, scalar_t(1.0), &ws.mu);
    solver::ineq_soft::bind_and_invalidate(&resto_box);
    solver::ineq_soft::bind_and_invalidate(&resto_stacked);
    resto_box.update_approximation(node_data::update_mode::eval_all, true);
    resto_stacked.update_approximation(node_data::update_mode::eval_all, true);

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        auto &ub = d.elastic.side[box_side::ub];
        auto &lb = d.elastic.side[box_side::lb];
        ub.value[detail::slot_t] << 1.2, 1.4;
        ub.dual[detail::slot_t] << 0.6, 0.5;
        ub.value[detail::slot_p] << 0.7, 0.8;
        ub.value[detail::slot_n] << 0.9, 1.1;
        ub.dual[detail::slot_p] << 1.0, 0.9;
        ub.dual[detail::slot_n] << 0.8, 0.7;
        lb.value[detail::slot_t] << 1.3, 1.5;
        lb.dual[detail::slot_t] << 0.55, 0.65;
        lb.value[detail::slot_p] << 0.6, 0.9;
        lb.value[detail::slot_n] << 1.0, 1.2;
        lb.dual[detail::slot_p] << 0.95, 0.85;
        lb.dual[detail::slot_n] << 0.75, 0.8;
    });
    resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        auto &side = d.elastic.side[box_side::ub];
        side.value[detail::slot_t] << 1.2, 1.3, 1.4, 1.5;
        side.dual[detail::slot_t] << 0.6, 0.55, 0.5, 0.65;
        side.value[detail::slot_p] << 0.7, 0.6, 0.8, 0.9;
        side.value[detail::slot_n] << 0.9, 1.0, 1.1, 1.2;
        side.dual[detail::slot_p] << 1.0, 0.95, 0.9, 0.85;
        side.dual[detail::slot_n] << 0.8, 0.75, 0.7, 0.8;
    });

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        resto_ineq_elastic_ipm_constr::compute_local_model(
            d.elastic, d.require_box_spec("test"), scalar_t(20.0), ws.mu);
    });
    resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        resto_ineq_elastic_ipm_constr::compute_local_model(
            d.elastic, d.require_box_spec("test"), scalar_t(20.0), ws.mu);
    });

    sync_restoration_to_outer_state(resto_box, outer_box);
    sync_restoration_to_outer_state(resto_stacked, outer_stacked);
    outer_box.update_approximation(node_data::update_mode::eval_val, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_val, true);

    scalar_t box_l1 = 0.;
    scalar_t box_inf = 0.;
    scalar_t stacked_l1 = 0.;
    scalar_t stacked_inf = 0.;
    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d) {
        box_l1 = d.v_.lpNorm<1>();
        box_inf = d.v_.size() > 0 ? d.v_.cwiseAbs().maxCoeff() : scalar_t(0.);
        REQUIRE((d.box_side_[box_side::ub]->slack - vector((vector(2) << 1.2, 1.4).finished())).cwiseAbs().maxCoeff() < 1e-12);
        REQUIRE((d.box_side_[box_side::lb]->slack - vector((vector(2) << 1.3, 1.5).finished())).cwiseAbs().maxCoeff() < 1e-12);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d) {
        vector stacked_view(2);
        stacked_view << std::max(d.v_(0), d.v_(1)), std::max(d.v_(2), d.v_(3));
        stacked_l1 = stacked_view.lpNorm<1>();
        stacked_inf = stacked_view.size() > 0 ? stacked_view.cwiseAbs().maxCoeff() : scalar_t(0.);
    });

    REQUIRE(approx_scalar(box_l1, stacked_l1, 1e-12));
    REQUIRE(approx_scalar(box_inf, stacked_inf, 1e-12));

    const scalar_t alpha_dual = scalar_t(0.4);
    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &c, solver::ipm_constr::approx_data &d) {
        c.restoration_commit_dual_step(d, alpha_dual);
    });
    outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &c, solver::ipm_constr::approx_data &d) {
        c.restoration_commit_dual_step(d, alpha_dual);
    });

    outer_box.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d_box) {
        outer_stacked.for_each(__ineq_xu, [&](const solver::ipm_constr &, solver::ipm_constr::approx_data &d_stacked) {
            REQUIRE(approx_scalar(d_box.multiplier_(0), d_stacked.multiplier_(0) - d_stacked.multiplier_(1), 1e-12));
            REQUIRE(approx_scalar(d_box.multiplier_(1), d_stacked.multiplier_(2) - d_stacked.multiplier_(3), 1e-12));
            REQUIRE((d_box.box_side_[box_side::ub]->multiplier -
                     vector((vector(2) << d_stacked.box_side_[box_side::ub]->multiplier(0),
                                     d_stacked.box_side_[box_side::ub]->multiplier(2)).finished()))
                        .cwiseAbs()
                        .maxCoeff() < 1e-12);
            REQUIRE((d_box.box_side_[box_side::lb]->multiplier -
                     vector((vector(2) << d_stacked.box_side_[box_side::ub]->multiplier(1),
                                     d_stacked.box_side_[box_side::ub]->multiplier(3)).finished()))
                        .cwiseAbs()
                        .maxCoeff() < 1e-12);
        });
    });
}

TEST_CASE("restoration boxed inequality runtime matches equivalent stacked rows through Newton and affine steps") {
    auto build_box_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_runtime_box", 2);
        prob->add(*u);
        const vector lb = (vector(2) << -1.0, -2.0).finished();
        const vector ub = (vector(2) << 3.0, 4.0).finished();
        prob->add(*ineq_constr::create(
            "u_box_runtime",
            var_inarg_list{*u},
            static_cast<const cs::SX &>(u),
            lb,
            ub));
        prob->wait_until_ready();
        return prob;
    };
    auto build_stacked_prob = []() {
        auto prob = ocp::create();
        auto u = sym::inputs("u_runtime_stacked", 2);
        prob->add(*u);
        const cs::SX g = cs::SX::vertcat(std::vector<cs::SX>{
            u(0) - scalar_t(3.0),
            scalar_t(-1.0) - u(0),
            u(1) - scalar_t(4.0),
            scalar_t(-2.0) - u(1),
        });
        prob->add(*ineq_constr::create("u_stacked_runtime", var_inarg_list{*u}, g));
        prob->wait_until_ready();
        return prob;
    };

    const auto box_prob = build_box_prob();
    const auto stacked_prob = build_stacked_prob();
    const auto resto_box_prob = build_restoration_overlay_problem(box_prob, restoration_overlay_settings{.rho_ineq = 20.0});
    const auto resto_stacked_prob =
        build_restoration_overlay_problem(stacked_prob, restoration_overlay_settings{.rho_ineq = 20.0});

    ns_sqp::data outer_box(box_prob);
    ns_sqp::data resto_box(resto_box_prob);
    ns_sqp::data outer_stacked(stacked_prob);
    ns_sqp::data resto_stacked(resto_stacked_prob);
    ns_sqp::settings_t ws;
    ws.mu = scalar_t(0.3);
    ws.alpha_primal = scalar_t(0.4);
    ws.alpha_dual = scalar_t(0.35);

    auto setup_stage = [&](ns_sqp::data &stage) {
        stage.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
            c.setup_workspace_data(fd, &ws);
        });
        solver::ineq_soft::bind_runtime(&stage);
    };
    setup_stage(outer_box);
    setup_stage(resto_box);
    setup_stage(outer_stacked);
    setup_stage(resto_stacked);

    outer_box.sym_val().value_[__u] << 0.25, -0.75;
    outer_stacked.sym_val().value_[__u] = outer_box.sym_val().value_[__u];
    outer_box.update_approximation(node_data::update_mode::eval_all, true);
    outer_stacked.update_approximation(node_data::update_mode::eval_all, true);

    sync_outer_to_restoration_state(outer_box, resto_box, scalar_t(1.0), &ws.mu);
    sync_outer_to_restoration_state(outer_stacked, resto_stacked, scalar_t(1.0), &ws.mu);
    solver::ineq_soft::bind_and_invalidate(&resto_box);
    solver::ineq_soft::bind_and_invalidate(&resto_stacked);
    resto_box.update_approximation(node_data::update_mode::eval_all, true);
    resto_stacked.update_approximation(node_data::update_mode::eval_all, true);

    resto_box.trial_prim_step[__u] << 0.2, -0.15;
    resto_stacked.trial_prim_step[__u] = resto_box.trial_prim_step[__u];

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        auto &ub = d.elastic.side[box_side::ub];
        auto &lb = d.elastic.side[box_side::lb];
        ub.value[detail::slot_t] << 1.2, 1.4;
        ub.dual[detail::slot_t] << 0.6, 0.5;
        ub.value[detail::slot_p] << 0.7, 0.8;
        ub.value[detail::slot_n] << 0.9, 1.1;
        ub.dual[detail::slot_p] << 1.0, 0.9;
        ub.dual[detail::slot_n] << 0.8, 0.7;
        lb.value[detail::slot_t] << 1.3, 1.5;
        lb.dual[detail::slot_t] << 0.55, 0.65;
        lb.value[detail::slot_p] << 0.6, 0.9;
        lb.value[detail::slot_n] << 1.0, 1.2;
        lb.dual[detail::slot_p] << 0.95, 0.85;
        lb.dual[detail::slot_n] << 0.75, 0.8;
        resto_ineq_elastic_ipm_constr::compute_local_model(
            d.elastic, d.require_box_spec("test"), scalar_t(20.0), ws.mu);
    });
    resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        auto &side = d.elastic.side[box_side::ub];
        side.value[detail::slot_t] << 1.2, 1.3, 1.4, 1.5;
        side.dual[detail::slot_t] << 0.6, 0.55, 0.5, 0.65;
        side.value[detail::slot_p] << 0.7, 0.6, 0.8, 0.9;
        side.value[detail::slot_n] << 0.9, 1.0, 1.1, 1.2;
        side.dual[detail::slot_p] << 1.0, 0.95, 0.9, 0.85;
        side.dual[detail::slot_n] << 0.8, 0.75, 0.7, 0.8;
        resto_ineq_elastic_ipm_constr::compute_local_model(
            d.elastic, d.require_box_spec("test"), scalar_t(20.0), ws.mu);
    });

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        overlay.finalize_newton_step(d);
    });
    resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        overlay.finalize_newton_step(d);
    });

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        const auto &ub = d.elastic.side[box_side::ub];
        const auto &lb = d.elastic.side[box_side::lb];
        resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &s) {
            const auto &side = s.elastic.side[box_side::ub];
            REQUIRE((ub.d_dual[detail::slot_t] - gather_entries(side.d_dual[detail::slot_t], {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
            REQUIRE((lb.d_dual[detail::slot_t] - gather_entries(side.d_dual[detail::slot_t], {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            for (auto slot : std::array{detail::slot_t, detail::slot_p, detail::slot_n}) {
                REQUIRE((ub.d_value[slot] - gather_entries(side.d_value[slot], {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
                REQUIRE((lb.d_value[slot] - gather_entries(side.d_value[slot], {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            }
        });
    });

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        overlay.apply_affine_step(d, &ws);
    });
    resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &overlay, resto_ineq_elastic_ipm_constr::approx_data &d) {
        overlay.apply_affine_step(d, &ws);
    });

    resto_box.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &d) {
        const auto &ub = d.elastic.side[box_side::ub];
        const auto &lb = d.elastic.side[box_side::lb];
        resto_stacked.for_each(__ineq_xu, [&](const resto_ineq_elastic_ipm_constr &, resto_ineq_elastic_ipm_constr::approx_data &s) {
            const auto &side = s.elastic.side[box_side::ub];
            for (auto slot : std::array{detail::slot_t, detail::slot_p, detail::slot_n}) {
                REQUIRE((ub.value[slot] - gather_entries(side.value[slot], {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
                REQUIRE((lb.value[slot] - gather_entries(side.value[slot], {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
                REQUIRE((ub.dual[slot] - gather_entries(side.dual[slot], {0, 2})).cwiseAbs().maxCoeff() < 1e-12);
                REQUIRE((lb.dual[slot] - gather_entries(side.dual[slot], {1, 3})).cwiseAbs().maxCoeff() < 1e-12);
            }
            REQUIRE((d.elastic.primal_view - vector((vector(2) << std::max(s.elastic.side[box_side::ub].r_d(0),
                                                                            s.elastic.side[box_side::ub].r_d(1)),
                                                         std::max(s.elastic.side[box_side::ub].r_d(2),
                                                                  s.elastic.side[box_side::ub].r_d(3)))
                                                       .finished()))
                        .cwiseAbs()
                        .maxCoeff() < 1e-12);
        });
    });
}

TEST_CASE("positivity helper reuses consistent alpha and backup semantics") {
    vector primal(3), primal_step(3), primal_backup(3);
    primal << 1.0, 2.0, 0.5;
    primal_step << -0.2, 0.1, -0.4;

    vector dual(3), dual_step(3), dual_backup(3);
    dual << 0.8, 1.5, 0.9;
    dual_step << -0.1, 0.2, -0.3;

    solver::linesearch_config cfg;
    positivity::update_pair_bounds(cfg, primal, primal_step, dual, dual_step);
    REQUIRE(approx_scalar(cfg.primal.alpha_max, std::min(1.0, -0.995 * 0.5 / -0.4)));
    REQUIRE(approx_scalar(cfg.dual.alpha_max, std::min(1.0, -0.995 * 0.9 / -0.3)));

    positivity::backup_pair(primal, primal_backup, dual, dual_backup);
    positivity::apply_pair_step(primal, primal_step, 0.5, dual, dual_step, 0.25);
    REQUIRE(approx_scalar(primal(0), 0.9));
    REQUIRE(approx_scalar(dual(2), 0.825));

    positivity::restore_pair(primal, primal_backup, dual, dual_backup);
    REQUIRE(approx_scalar(primal(0), 1.0));
    REQUIRE(approx_scalar(dual(2), 0.9));
}

TEST_CASE("restoration elastic blocks own their penalty and barrier bookkeeping") {
    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 1);
    slot_value(eq, detail::slot_p) << 2.0, 3.0;
    slot_value(eq, detail::slot_n) << 5.0, 7.0;
    slot_value_backup(eq, detail::slot_p) << 4.0, 8.0;
    slot_value_backup(eq, detail::slot_n) << 2.0, 10.0;
    slot_step(eq, detail::slot_p) << 0.5, -1.0;
    slot_step(eq, detail::slot_n) << 1.0, 2.0;

    REQUIRE(approx_scalar(slot_value(eq, detail::slot_p).sum() + slot_value(eq, detail::slot_n).sum(), 17.0));
    REQUIRE(approx_scalar(slot_step(eq, detail::slot_p).sum() + slot_step(eq, detail::slot_n).sum(), 2.5));
    REQUIRE(approx_scalar(slot_value(eq, detail::slot_p).array().log().sum() + slot_value(eq, detail::slot_n).array().log().sum(),
                          std::log(2.0) + std::log(3.0) + std::log(5.0) + std::log(7.0)));
    REQUIRE(approx_scalar((slot_step(eq, detail::slot_p).array() / slot_value_backup(eq, detail::slot_p).array()).sum() +
                              (slot_step(eq, detail::slot_n).array() / slot_value_backup(eq, detail::slot_n).array()).sum(),
                          0.5 / 4.0 - 1.0 / 8.0 + 1.0 / 2.0 + 2.0 / 10.0));

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 1);
    slot_value(iq, detail::slot_t) << 11.0, 13.0;
    slot_value(iq, detail::slot_p) << 17.0, 19.0;
    slot_value(iq, detail::slot_n) << 23.0, 29.0;
    slot_value_backup(iq, detail::slot_t) << 31.0, 37.0;
    slot_value_backup(iq, detail::slot_p) << 41.0, 43.0;
    slot_value_backup(iq, detail::slot_n) << 47.0, 53.0;
    slot_step(iq, detail::slot_t) << -2.0, 1.0;
    slot_step(iq, detail::slot_p) << 3.0, 4.0;
    slot_step(iq, detail::slot_n) << -5.0, 6.0;

    REQUIRE(approx_scalar(slot_value(iq, detail::slot_p).sum() + slot_value(iq, detail::slot_n).sum(), 88.0));
    REQUIRE(approx_scalar(slot_step(iq, detail::slot_p).sum() + slot_step(iq, detail::slot_n).sum(), 8.0));
    REQUIRE(approx_scalar(slot_value(iq, detail::slot_t).array().log().sum() +
                              slot_value(iq, detail::slot_p).array().log().sum() +
                              slot_value(iq, detail::slot_n).array().log().sum(),
                          std::log(11.0) + std::log(13.0) + std::log(17.0) + std::log(19.0) + std::log(23.0) + std::log(29.0)));
    REQUIRE(approx_scalar((slot_step(iq, detail::slot_t).array() / slot_value_backup(iq, detail::slot_t).array()).sum() +
                              (slot_step(iq, detail::slot_p).array() / slot_value_backup(iq, detail::slot_p).array()).sum() +
                              (slot_step(iq, detail::slot_n).array() / slot_value_backup(iq, detail::slot_n).array()).sum(),
                          -2.0 / 31.0 + 1.0 / 37.0 + 3.0 / 41.0 + 4.0 / 43.0 - 5.0 / 47.0 + 6.0 / 53.0));
}

TEST_CASE("restoration predictor bookkeeping follows normal IPM complementarity accounting") {
    solver::linesearch_config cfg;
    cfg.alpha_primal = 0.4;
    cfg.alpha_dual = 0.7;

    detail::eq_local_state eq;
    resize_eq_state(eq, 1, 0);
    slot_value(eq, detail::slot_p) << 2.0;
    slot_value(eq, detail::slot_n) << 3.0;
    slot_dual(eq, detail::slot_p) << 5.0;
    slot_dual(eq, detail::slot_n) << 7.0;
    slot_step(eq, detail::slot_p) << -0.5;
    slot_step(eq, detail::slot_n) << 0.25;
    slot_dual_step(eq, detail::slot_p) << -1.5;
    slot_dual_step(eq, detail::slot_n) << 0.5;

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 0);
    slot_value(iq, detail::slot_t) << 11.0;
    slot_value(iq, detail::slot_p) << 13.0;
    slot_value(iq, detail::slot_n) << 17.0;
    slot_dual(iq, detail::slot_t) << 19.0;
    slot_dual(iq, detail::slot_p) << 23.0;
    slot_dual(iq, detail::slot_n) << 29.0;
    slot_step(iq, detail::slot_t) << 0.5;
    slot_step(iq, detail::slot_p) << 1.5;
    slot_step(iq, detail::slot_n) << -1.0;
    slot_dual_step(iq, detail::slot_t) << -3.0;
    slot_dual_step(iq, detail::slot_p) << 2.0;
    slot_dual_step(iq, detail::slot_n) << -4.0;

    solver::ipm_config::worker worker;
    const scalar_t alpha_d = cfg.dual_alpha_for_ineq();
    for (const auto &[value, step, dual, dual_step] : {
             std::tuple{&slot_value(eq, detail::slot_p), &slot_step(eq, detail::slot_p),
                        &slot_dual(eq, detail::slot_p), &slot_dual_step(eq, detail::slot_p)},
             std::tuple{&slot_value(eq, detail::slot_n), &slot_step(eq, detail::slot_n),
                        &slot_dual(eq, detail::slot_n), &slot_dual_step(eq, detail::slot_n)},
             std::tuple{&slot_value(iq, detail::slot_t), &slot_step(iq, detail::slot_t),
                        &slot_dual(iq, detail::slot_t), &slot_dual_step(iq, detail::slot_t)},
             std::tuple{&slot_value(iq, detail::slot_p), &slot_step(iq, detail::slot_p),
                        &slot_dual(iq, detail::slot_p), &slot_dual_step(iq, detail::slot_p)},
             std::tuple{&slot_value(iq, detail::slot_n), &slot_step(iq, detail::slot_n),
                        &slot_dual(iq, detail::slot_n), &slot_dual_step(iq, detail::slot_n)},
         }) {
        worker.n_ipm_cstr += static_cast<size_t>(value->size());
        worker.prev_aff_comp += dual->dot(*value);
        worker.post_aff_comp +=
            (*dual + alpha_d * *dual_step).dot(*value + cfg.alpha_primal * *step);
    }

    REQUIRE(worker.n_ipm_cstr == 5);
    const scalar_t prev_aff =
        2.0 * 5.0 + 3.0 * 7.0 + 11.0 * 19.0 + 13.0 * 23.0 + 17.0 * 29.0;
    const scalar_t post_aff =
        (5.0 + alpha_d * -1.5) * (2.0 + cfg.alpha_primal * -0.5) +
        (7.0 + alpha_d * 0.5) * (3.0 + cfg.alpha_primal * 0.25) +
        (19.0 + alpha_d * -3.0) * (11.0 + cfg.alpha_primal * 0.5) +
        (23.0 + alpha_d * 2.0) * (13.0 + cfg.alpha_primal * 1.5) +
        (29.0 + alpha_d * -4.0) * (17.0 + cfg.alpha_primal * -1.0);
    REQUIRE(approx_scalar(worker.prev_aff_comp, prev_aff));
    REQUIRE(approx_scalar(worker.post_aff_comp, post_aff));
}
