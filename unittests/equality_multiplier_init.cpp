#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

using namespace moto;

cost make_stage_cost(const std::string &name, const sym &x, const sym &u) {
    auto c = cost(new generic_cost(name, approx_order::second));
    dynamic_cast<generic_func &>(*c).add_argument(x);
    dynamic_cast<generic_func &>(*c).add_argument(u);
    c->value = [](func_approx_data &d) {
        d.v_(0) += scalar_t(0.5) * (scalar_t(2.0) * d[0].squaredNorm() + scalar_t(3.0) * d[1].squaredNorm());
    };
    c->jacobian = [](func_approx_data &d) {
        d.jac_[0].noalias() += scalar_t(2.0) * d[0].transpose();
        d.jac_[1].noalias() += scalar_t(3.0) * d[1].transpose();
    };
    c->hessian = [](func_approx_data &d) {
        d.lag_hess_[0][0].diagonal().array() += scalar_t(2.0);
        d.lag_hess_[1][1].diagonal().array() += scalar_t(3.0);
    };
    return c;
}

constr make_hard_eq_x(const std::string &name, const sym &x, scalar_t target) {
    auto c = constr(new generic_constr(name, approx_order::first, 1));
    c->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*c).add_argument(x);
    c->value = [target](func_approx_data &d) { d.v_(0) = d[0](0) - target; };
    c->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    return c;
}

constr make_hard_eq_xu(const std::string &name, const sym &x, const sym &u, scalar_t bias) {
    auto c = constr(new generic_constr(name, approx_order::first, 1));
    c->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*c).add_argument(x);
    dynamic_cast<generic_func &>(*c).add_argument(u);
    c->value = [bias](func_approx_data &d) { d.v_(0) = d[0](0) + scalar_t(2.0) * d[1](0) + bias; };
    c->jacobian = [](func_approx_data &d) {
        d.jac_[0](0, 0) = 1.;
        d.jac_[1](0, 0) = 2.;
    };
    return c;
}

constr make_soft_eq_x(const std::string &name, const sym &x, scalar_t target) {
    auto base = constr(new generic_constr(name, approx_order::first, 1));
    base->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*base).add_argument(x);
    base->value = [target](func_approx_data &d) { d.v_(0) = d[0](0) - target; };
    base->jacobian = [](func_approx_data &d) { d.jac_[0](0, 0) = 1.; };
    return constr(base->cast_soft("pmm_constr"));
}

constr make_soft_eq_xu(const std::string &name, const sym &x, const sym &u, scalar_t bias) {
    auto base = constr(new generic_constr(name, approx_order::first, 1));
    base->field_hint().is_eq = true;
    dynamic_cast<generic_func &>(*base).add_argument(x);
    dynamic_cast<generic_func &>(*base).add_argument(u);
    base->value = [bias](func_approx_data &d) { d.v_(0) = scalar_t(-0.5) * d[0](0) + d[1](0) + bias; };
    base->jacobian = [](func_approx_data &d) {
        d.jac_[0](0, 0) = scalar_t(-0.5);
        d.jac_[1](0, 0) = 1.;
    };
    return constr(base->cast_soft("pmm_constr"));
}

constr make_ineq_xu(const std::string &name, const sym &x, const sym &u, scalar_t bias) {
    auto base = constr(new generic_constr(name, approx_order::first, 1));
    base->field_hint().is_eq = false;
    dynamic_cast<generic_func &>(*base).add_argument(x);
    dynamic_cast<generic_func &>(*base).add_argument(u);
    base->value = [bias](func_approx_data &d) { d.v_(0) = d[0](0) - d[1](0) + bias; };
    base->jacobian = [](func_approx_data &d) {
        d.jac_[0](0, 0) = 1.;
        d.jac_[1](0, 0) = -1.;
    };
    return constr(base->cast_ineq("ipm"));
}

void seed_primal_state(ns_sqp &sqp, size_t n_stage_nodes) {
    auto &flat = sqp.active_data().flatten_nodes();
    REQUIRE(flat.size() == n_stage_nodes);

    for (size_t i = 0; i < flat.size(); ++i) {
        flat[i]->sym_val().value_[__x](0) = scalar_t(0.8 - 0.35 * static_cast<scalar_t>(i));
        flat[i]->sym_val().value_[__u](0) = scalar_t(-0.35 + 0.4 * static_cast<scalar_t>(i));
        flat[i]->sym_val().value_[__y](0) = scalar_t(0.55 - 0.25 * static_cast<scalar_t>(i));
    }
}

void configure_solver(ns_sqp &sqp, bool enable_eq_init, size_t n_edges) {
    auto [x, y] = sym::states("x_eq_init", 1);
    auto u = sym::inputs("u_eq_init", 1);
    const var_inarg_list dyn_args = var_list{x, y, u};
    auto dyn = dynamics(new dense_dynamics("dyn_eq_init", dyn_args, y - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*make_stage_cost("stage_cost_eq_init", x, u));
    stage_prob->add(*make_hard_eq_x("hard_eq_x_eq_init", x, scalar_t(0.25)));
    stage_prob->add(*make_hard_eq_xu("hard_eq_xu_eq_init", x, u, scalar_t(-0.1)));
    stage_prob->add(*make_soft_eq_x("soft_eq_x_eq_init", x, scalar_t(-0.2)));
    stage_prob->add(*make_soft_eq_xu("soft_eq_xu_eq_init", x, u, scalar_t(0.05)));
    stage_prob->add(*make_ineq_xu("ineq_xu_eq_init", x, u, scalar_t(-0.3)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("terminal_cost_eq_init", var_list{x}, x * x, approx_order::second)));

    sqp.settings.no_except = false;
    sqp.settings.restoration.enabled = false;
    sqp.settings.eq_init.enabled = enable_eq_init;
    sqp.settings.eq_init.rho_eq = 10.0;

    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto edges = modeled.add_path(n0, nt, n_edges);
    for (const auto &edge : edges) {
        edge->add(*dyn);
    }

    seed_primal_state(sqp, n_edges);
}
} // namespace

TEST_CASE("equality multiplier initialization leaves primals and inequalities fixed") {
    ns_sqp without_init;
    ns_sqp with_init;
    configure_solver(without_init, false, 1);
    configure_solver(with_init, true, 1);

    REQUIRE_NOTHROW(without_init.update(0, false));
    REQUIRE_NOTHROW(with_init.update(0, false));

    auto &flat_without = without_init.active_data().flatten_nodes();
    auto &flat_with = with_init.active_data().flatten_nodes();
    REQUIRE(flat_without.size() == flat_with.size());

    bool saw_soft_dual = false;
    for (size_t i = 0; i < flat_with.size(); ++i) {
        auto *lhs = flat_without[i];
        auto *rhs = flat_with[i];

        REQUIRE(rhs->sym_val().value_[__x].isApprox(lhs->sym_val().value_[__x]));
        REQUIRE(rhs->sym_val().value_[__u].isApprox(lhs->sym_val().value_[__u]));
        REQUIRE(rhs->sym_val().value_[__y].isApprox(lhs->sym_val().value_[__y]));
        REQUIRE(rhs->sym_val().value_[__s].isApprox(lhs->sym_val().value_[__s]));

        REQUIRE(rhs->dense().dual_[__ineq_x].isApprox(lhs->dense().dual_[__ineq_x]));
        REQUIRE(rhs->dense().dual_[__ineq_xu].isApprox(lhs->dense().dual_[__ineq_xu]));

        saw_soft_dual = saw_soft_dual ||
                        rhs->dense().dual_[__eq_x_soft].size() > 0 ||
                        rhs->dense().dual_[__eq_xu_soft].size() > 0;
    }

    REQUIRE(saw_soft_dual);
}

TEST_CASE("equality multiplier initialization reduces initial hard-equality dual residual on a two-stage chain") {
    ns_sqp without_init;
    ns_sqp with_init;
    configure_solver(without_init, false, 2);
    configure_solver(with_init, true, 2);

    const auto kkt_without = without_init.update(0, false);
    const auto kkt_with = with_init.update(0, false);

    auto &flat_without = without_init.active_data().flatten_nodes();
    auto &flat_with = with_init.active_data().flatten_nodes();
    REQUIRE(flat_without.size() == flat_with.size());

    bool changed_hard_dual = false;
    for (size_t i = 0; i < flat_with.size(); ++i) {
        changed_hard_dual = changed_hard_dual ||
                            !flat_with[i]->dense().dual_[__dyn].isApprox(flat_without[i]->dense().dual_[__dyn]) ||
                            !flat_with[i]->dense().dual_[__eq_x].isApprox(flat_without[i]->dense().dual_[__eq_x]) ||
                            !flat_with[i]->dense().dual_[__eq_xu].isApprox(flat_without[i]->dense().dual_[__eq_xu]);
    }

    REQUIRE(changed_hard_dual);
    REQUIRE(kkt_with.dual.inf_res < kkt_without.dual.inf_res);
}

TEST_CASE("equality multiplier initialization updates soft equalities and leaves inequality state unchanged") {
    ns_sqp without_init;
    ns_sqp with_init;
    configure_solver(without_init, false, 2);
    configure_solver(with_init, true, 2);

    REQUIRE_NOTHROW(without_init.update(0, false));
    REQUIRE_NOTHROW(with_init.update(0, false));

    auto &flat_without = without_init.active_data().flatten_nodes();
    auto &flat_with = with_init.active_data().flatten_nodes();
    REQUIRE(flat_without.size() == flat_with.size());

    bool saw_soft_dual = false;
    for (size_t i = 0; i < flat_with.size(); ++i) {
        saw_soft_dual = saw_soft_dual ||
                        flat_with[i]->dense().dual_[__eq_x_soft].size() > 0 ||
                        flat_with[i]->dense().dual_[__eq_xu_soft].size() > 0;

        REQUIRE(flat_with[i]->dense().dual_[__ineq_x].isApprox(flat_without[i]->dense().dual_[__ineq_x]));
        REQUIRE(flat_with[i]->dense().dual_[__ineq_xu].isApprox(flat_without[i]->dense().dual_[__ineq_xu]));
        REQUIRE(flat_with[i]->sym_val().value_[__s].isApprox(flat_without[i]->sym_val().value_[__s]));
        if (flat_with[i]->dense().dual_[__eq_x_soft].size() > 0) {
            REQUIRE(flat_with[i]->dense().dual_[__eq_x_soft].allFinite());
        }
        if (flat_with[i]->dense().dual_[__eq_xu_soft].size() > 0) {
            REQUIRE(flat_with[i]->dense().dual_[__eq_xu_soft].allFinite());
        }
    }

    REQUIRE(saw_soft_dual);
}
