#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <cmath>

#include <moto/ocp/impl/node_data.hpp>
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

template <typename State>
vector &slot_value(State &state, detail::elastic_slot_t slot) {
    return state.value[slot];
}

template <typename State>
const vector &slot_value(const State &state, detail::elastic_slot_t slot) {
    return state.value[slot];
}

template <typename State>
vector &slot_value_backup(State &state, detail::elastic_slot_t slot) {
    return state.value_backup[slot];
}

template <typename State>
vector &slot_step(State &state, detail::elastic_slot_t slot) {
    return state.d_value[slot];
}

template <typename State>
const vector &slot_step(const State &state, detail::elastic_slot_t slot) {
    return state.d_value[slot];
}

template <typename State>
vector &slot_dual(State &state, detail::elastic_slot_t slot) {
    return state.dual[slot];
}

template <typename State>
const vector &slot_dual(const State &state, detail::elastic_slot_t slot) {
    return state.dual[slot];
}

template <typename State>
vector &slot_dual_backup(State &state, detail::elastic_slot_t slot) {
    return state.dual_backup[slot];
}

template <typename State>
vector &slot_dual_step(State &state, detail::elastic_slot_t slot) {
    return state.d_dual[slot];
}

template <typename State>
const vector &slot_dual_step(const State &state, detail::elastic_slot_t slot) {
    return state.d_dual[slot];
}

template <typename State>
vector &slot_r_stat(State &state, detail::elastic_slot_t slot) {
    return state.r_stat[slot];
}

template <typename State>
const vector &slot_r_stat(const State &state, detail::elastic_slot_t slot) {
    return state.r_stat[slot];
}

template <typename State>
vector &slot_r_comp(State &state, detail::elastic_slot_t slot) {
    return state.r_comp[slot];
}

template <typename State>
const vector &slot_r_comp(const State &state, detail::elastic_slot_t slot) {
    return state.r_comp[slot];
}

template <typename State>
vector &slot_denom(State &state, detail::elastic_slot_t slot) {
    return state.denom[slot];
}

template <typename State>
vector &slot_backsub_rhs(State &state, detail::elastic_slot_t slot) {
    return state.backsub_rhs[slot];
}

template <typename State>
vector &slot_corrector(State &state, detail::elastic_slot_t slot) {
    return state.corrector[slot];
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
    resto_eq_elastic_constr::recover_local_step(delta_c, eq);

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

    vector mu_p = vector::Zero(2);
    vector mu_n = vector::Zero(2);
    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar, &mu_p, &mu_n);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    resto_eq_elastic_constr::recover_local_step(delta_c, eq);

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

    vector mu_p(2), mu_n(2);
    mu_p << 0.25, 0.28;
    mu_n << 0.27, 0.29;
    resto_eq_elastic_constr::compute_local_model(eq, c, lambda, rho, mu_bar, &mu_p, &mu_n);

    vector delta_c(2);
    delta_c << -0.11, 0.07;
    resto_eq_elastic_constr::recover_local_step(delta_c, eq);

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
        d.lag_jac_[0].noalias() += d[0].transpose();
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
    REQUIRE(eq_overlay->source_field() == __eq_x_soft);

    ns_sqp::data outer(prob);
    ns_sqp::data resto(resto_prob);
    ns_sqp::settings_t ws;

    resto.for_each_constr([&](const generic_func &c, func_approx_data &fd) {
        c.setup_workspace_data(fd, &ws);
    });

    REQUIRE(outer.dense().dual_[__eq_x_soft].size() == 1);
    outer.dense().dual_[__eq_x_soft](0) = scalar_t(2.5);

    sync_restoration_overlay_primal(outer, resto);
    sync_restoration_overlay_duals(outer, resto);
    seed_restoration_overlay_refs(resto, scalar_t(1.0));

    bool saw_overlay = false;
    resto.for_each(__eq_x_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        saw_overlay = true;
        REQUIRE(overlay.source()->name() == soft_eq->name());
        REQUIRE(approx_scalar(d.multiplier_(0), scalar_t(2.5)));
    });
    REQUIRE(saw_overlay);

    resto.update_approximation(node_data::update_mode::eval_val, true);
    solver::ineq_soft::initialize(&resto);

    saw_overlay = false;
    resto.for_each(__eq_x_soft, [&](const resto_eq_elastic_constr &overlay, resto_eq_elastic_constr::approx_data &d) {
        saw_overlay = true;
        REQUIRE(overlay.source()->name() == soft_eq->name());
        REQUIRE(approx_scalar(d.multiplier_(0), scalar_t(2.5)));
        REQUIRE(approx_scalar(d.multiplier_backup(0), scalar_t(2.5)));
    });
    REQUIRE(saw_overlay);
}

TEST_CASE("box helper lowers to doubled one-sided inequality rows") {
    auto [x, _y] = sym::states("x_box", 2);
    const vector lb = (vector(2) << -1.0, 0.5).finished();
    const vector ub = (vector(2) << 2.0, 3.0).finished();

    auto box = generic_constr::create_box("x_box_bound", var_inarg_list{*x}, static_cast<const cs::SX &>(x), lb, ub);
    REQUIRE(box->dim() == 4);
    REQUIRE(box->field_hint().is_eq == utils::optional_bool::False);

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);
}

TEST_CASE("box helper uses affine fast path for single-arg linear selection") {
    auto [x, _y] = sym::states("x_sel", 5);
    const vector lb = (vector(2) << -1.0, 2.0).finished();
    const vector ub = (vector(2) << 4.0, 5.0).finished();

    auto bound = generic_constr::create_box(
        "x_sel_bound",
        var_inarg_list{*x},
        cs::SX::vertcat(std::vector<cs::SX>{x(0), x(3)}),
        lb,
        ub);
    REQUIRE(bound->dim() == 4);
    REQUIRE(bound->field_hint().is_eq == utils::optional_bool::False);
    REQUIRE(bound->get_codegen_task() == nullptr);

    REQUIRE(bound->finalize());
    REQUIRE(bound->field() == __ineq_x);
}

TEST_CASE("box helper drops unbounded sides row-wise") {
    auto [x, _y] = sym::states("x_half_box", 2);
    const vector lb = (vector(2) << -std::numeric_limits<scalar_t>::infinity(), 0.5).finished();
    const vector ub = (vector(2) << 2.0, std::numeric_limits<scalar_t>::infinity()).finished();

    auto box = generic_constr::create_box("x_half_box_bound", var_inarg_list{*x}, static_cast<const cs::SX &>(x), lb, ub);
    REQUIRE(box->dim() == 2);
    REQUIRE(box->field_hint().is_eq == utils::optional_bool::False);

    REQUIRE(box->finalize());
    REQUIRE(box->field() == __ineq_x);
}

TEST_CASE("box helper accepts symbolic parameter bounds through generic casadi lowering") {
    auto [x, _y] = sym::states("x_param_box", 2);
    auto p = sym::params("p_box", 2);

    auto box = generic_constr::create_box(
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

TEST_CASE("box helper rejects symbolic bounds whose dependencies are missing from in_args") {
    auto [x, _y] = sym::states("x_missing_box", 2);
    auto p = sym::params("p_missing_box", 2);

    REQUIRE_THROWS_WITH(
        generic_constr::create_box(
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
    auto box_base = generic_constr::create_box(
        "x_box_bound",
        var_inarg_list{*u},
        cs::SX::vertcat(std::vector<cs::SX>{u(0), scalar_t(2.0) * u(0)}),
        lb,
        ub);
    constr box(box_base->cast_ineq("ipm"));
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
    REQUIRE(ineq_overlay->source()->dim() == 4);
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

    resto_ineq_elastic_ipm_constr::compute_local_model(iq, g, rho, mu_bar);

    vector delta_g(3);
    delta_g << 0.08, -0.12, 0.04;
    resto_ineq_elastic_ipm_constr::recover_local_step(delta_g, iq);

    const vector res_d = delta_g + slot_step(iq, detail::slot_t) - slot_step(iq, detail::slot_p) +
                         slot_step(iq, detail::slot_n) + iq.r_d;
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

TEST_CASE("restoration inequality initialization centers local elastic KKT system") {
    const scalar_t g = -0.25;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.2;
    const scalar_t nu_t0 = 0.7;

    const auto init = resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(g, rho, mu_bar, nu_t0);

    REQUIRE(approx_scalar(init.nu_t, nu_t0));
    REQUIRE(init.t > 0.);
    REQUIRE(init.p > 0.);
    REQUIRE(init.n > 0.);
    REQUIRE(init.nu_p > 0.);
    REQUIRE(init.nu_n > 0.);
    REQUIRE(approx_scalar(init.nu_p, rho - init.nu_t, 1e-12));
    REQUIRE(approx_scalar(init.nu_n, rho + init.nu_t, 1e-12));
    REQUIRE(approx_scalar(g + init.t - init.p + init.n, 0.0, 1e-12));
    REQUIRE(approx_scalar(init.nu_p * init.p, mu_bar, 1e-10));
    REQUIRE(approx_scalar(init.nu_n * init.n, mu_bar, 1e-10));
    REQUIRE(approx_scalar(init.t * init.nu_t - mu_bar,
                          nu_t0 * (-g + init.p - init.n) - mu_bar,
                          1e-12));
}

TEST_CASE("restoration inequality initializer yields centered local model") {
    const scalar_t g = -0.35;
    const scalar_t rho = 3.0;
    const scalar_t mu_bar = 0.25;
    const scalar_t nu_t0 = 0.8;

    const auto init = resto_ineq_elastic_ipm_constr::initialize_elastic_ineq_scalar(g, rho, mu_bar, nu_t0);

    detail::ineq_local_state iq;
    resize_ineq_state(iq, 1, 0);
    slot_value(iq, detail::slot_t) << init.t;
    slot_value(iq, detail::slot_p) << init.p;
    slot_value(iq, detail::slot_n) << init.n;
    slot_dual(iq, detail::slot_t) << init.nu_t;
    slot_dual(iq, detail::slot_p) << init.nu_p;
    slot_dual(iq, detail::slot_n) << init.nu_n;

    vector g_vec(1);
    g_vec << g;
    resto_ineq_elastic_ipm_constr::compute_local_model(iq, g_vec, rho, mu_bar);

    REQUIRE(approx_zero(iq.r_d));
    REQUIRE(approx_zero(slot_r_comp(iq, detail::slot_p)));
    REQUIRE(approx_zero(slot_r_comp(iq, detail::slot_n)));
    REQUIRE(approx_zero(slot_r_stat(iq, detail::slot_p)));
    REQUIRE(approx_zero(slot_r_stat(iq, detail::slot_n)));
    REQUIRE(slot_r_comp(iq, detail::slot_t).allFinite());
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

TEST_CASE("restoration exit helper restores outer dual arrays") {
    array_type<vector, constr_fields> dual;
    array_type<vector, constr_fields> backup;

    scalar_t seed = 1.0;
    for (auto f : constr_fields) {
        dual[f] = vector::Constant(2, seed);
        backup[f] = vector::Constant(2, 10.0 + seed);
        seed += 1.0;
    }

    restore_outer_duals(dual, backup);

    for (auto f : constr_fields) {
        REQUIRE(dual[f].isApprox(backup[f]));
    }
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
