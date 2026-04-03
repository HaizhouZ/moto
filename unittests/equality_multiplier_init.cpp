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
        d.lag_jac_[0].noalias() += scalar_t(2.0) * d[0].transpose();
        d.lag_jac_[1].noalias() += scalar_t(3.0) * d[1].transpose();
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
    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == n_stage_nodes);

    for (size_t i = 0; i < flat.size(); ++i) {
        flat[i]->sym_val().value_[__x](0) = scalar_t(0.8 - 0.35 * static_cast<scalar_t>(i));
        flat[i]->sym_val().value_[__u](0) = scalar_t(-0.35 + 0.4 * static_cast<scalar_t>(i));
        flat[i]->sym_val().value_[__y](0) = scalar_t(0.55 - 0.25 * static_cast<scalar_t>(i));
    }
}

using primal_snapshot_t = std::array<std::vector<vector>, field::num_prim>;

primal_snapshot_t capture_primal_state(ns_sqp &sqp) {
    primal_snapshot_t snapshot;
    auto &flat = sqp.graph().flatten_nodes();
    for (auto field : primal_fields) {
        auto &field_snap = snapshot[field];
        field_snap.reserve(flat.size());
        for (auto *node : flat) {
            field_snap.push_back(node->sym_val().value_[field]);
        }
    }
    return snapshot;
}

void restore_primal_state(ns_sqp &sqp, const primal_snapshot_t &snapshot) {
    auto &flat = sqp.graph().flatten_nodes();
    for (auto field : primal_fields) {
        REQUIRE(snapshot[field].size() == flat.size());
        for (size_t i = 0; i < flat.size(); ++i) {
            flat[i]->sym_val().value_[field] = snapshot[field][i];
        }
    }
}

std::vector<vector> capture_equality_duals(ns_sqp &sqp) {
    std::vector<vector> snapshot;
    auto &flat = sqp.graph().flatten_nodes();
    snapshot.reserve(flat.size() * 5);
    for (auto *node : flat) {
        for (auto field : std::array{__dyn, __eq_x, __eq_xu, __eq_x_soft, __eq_xu_soft}) {
            snapshot.push_back(node->dense().dual_[field]);
        }
    }
    return snapshot;
}

scalar_t max_snapshot_diff(const std::vector<vector> &lhs, const std::vector<vector> &rhs) {
    REQUIRE(lhs.size() == rhs.size());
    scalar_t max_diff = 0.;
    for (size_t i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs[i].size() == rhs[i].size());
        if (lhs[i].size() == 0) {
            continue;
        }
        max_diff = std::max(max_diff, (lhs[i] - rhs[i]).cwiseAbs().maxCoeff());
    }
    return max_diff;
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

    auto &flat_without = without_init.graph().flatten_nodes();
    auto &flat_with = with_init.graph().flatten_nodes();
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

    auto &flat_without = without_init.graph().flatten_nodes();
    auto &flat_with = with_init.graph().flatten_nodes();
    REQUIRE(flat_without.size() == flat_with.size());

    bool changed_hard_dual = false;
    for (size_t i = 0; i < flat_with.size(); ++i) {
        changed_hard_dual = changed_hard_dual ||
                            !flat_with[i]->dense().dual_[__dyn].isApprox(flat_without[i]->dense().dual_[__dyn]) ||
                            !flat_with[i]->dense().dual_[__eq_x].isApprox(flat_without[i]->dense().dual_[__eq_x]) ||
                            !flat_with[i]->dense().dual_[__eq_xu].isApprox(flat_without[i]->dense().dual_[__eq_xu]);
    }

    REQUIRE(changed_hard_dual);
    REQUIRE(kkt_with.inf_dual_res < kkt_without.inf_dual_res);
}

TEST_CASE("equality multiplier initialization updates soft equalities and leaves inequality state unchanged") {
    ns_sqp without_init;
    ns_sqp with_init;
    configure_solver(without_init, false, 2);
    configure_solver(with_init, true, 2);

    REQUIRE_NOTHROW(without_init.update(0, false));
    REQUIRE_NOTHROW(with_init.update(0, false));

    auto &flat_without = without_init.graph().flatten_nodes();
    auto &flat_with = with_init.graph().flatten_nodes();
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

TEST_CASE("repeated fixed-primal equality multiplier rebuilds settle") {
    ns_sqp with_init;
    configure_solver(with_init, true, 2);
    with_init.settings.ipm.warm_start = true;

    const auto frozen_primal = capture_primal_state(with_init);

    std::vector<scalar_t> dual_residuals;
    std::vector<std::vector<vector>> dual_snapshots;
    constexpr size_t n_passes = 3;
    dual_residuals.reserve(n_passes);
    dual_snapshots.reserve(n_passes);

    for (size_t pass = 0; pass < n_passes; ++pass) {
        restore_primal_state(with_init, frozen_primal);
        const auto kkt = with_init.update(0, false);
        dual_residuals.push_back(kkt.inf_dual_res);
        dual_snapshots.push_back(capture_equality_duals(with_init));
    }

    const auto [res_min_it, res_max_it] = std::minmax_element(dual_residuals.begin(), dual_residuals.end());
    REQUIRE(*res_max_it - *res_min_it <= scalar_t(0.1) * dual_residuals.front());

    const scalar_t diff_12 = max_snapshot_diff(dual_snapshots[0], dual_snapshots[1]);
    const scalar_t diff_23 = max_snapshot_diff(dual_snapshots[1], dual_snapshots[2]);
    REQUIRE(diff_23 <= diff_12 + scalar_t(1e-12));
}
