#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <moto/model/graph_model.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
std::vector<std::string> expr_names(const moto::ocp_base_ptr_t &prob, moto::field_t field) {
    std::vector<std::string> names;
    for (const moto::shared_expr &expr : prob->exprs(field)) {
        names.push_back(expr->name());
    }
    return names;
}

bool contains_name(const std::vector<std::string> &names, const std::string &target) {
    return std::find(names.begin(), names.end(), target) != names.end();
}

const moto::generic_func &require_func_named(const moto::ocp_base_ptr_t &prob,
                                             moto::field_t field,
                                             const std::string &name) {
    auto it = std::find_if(prob->exprs(field).begin(), prob->exprs(field).end(), [&](const moto::shared_expr &expr) {
        return expr->name() == name;
    });
    REQUIRE(it != prob->exprs(field).end());
    const auto *func = dynamic_cast<const moto::generic_func *>((*it).get());
    REQUIRE(func != nullptr);
    return *func;
}
} // namespace

TEST_CASE("graph_model compose lowers only predecessor-owned eq_x terms") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x", 1);
    auto u = sym::inputs("u", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto make_node_prob = [&](const std::string &tag) {
        auto prob = node_ocp::create();
        auto eq = constr(new generic_constr("eq_pred_owned_" + tag, x_args, x, approx_order::second, __undefined));
        auto c = cost(new generic_cost("cost_pred_owned_" + tag, x_args, x, approx_order::second));
        prob->add(*eq);
        prob->add(*c);
        return prob;
    };

    graph_model modeled;
    auto n0 = modeled.add_node(make_node_prob("A"));
    auto n1 = modeled.add_node(make_node_prob("B"));
    auto n2 = modeled.add_node(make_node_prob("C"));

    auto e01 = modeled.connect(n0, n1);
    auto e12 = modeled.connect(n1, n2);
    e01->add(*dyn);
    e12->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 2);
    auto p01 = composed[0];
    auto p12 = composed[1];

    const auto eq01 = expr_names(p01, __eq_x);
    const auto eq12 = expr_names(p12, __eq_x);
    const auto cost01 = expr_names(p01, __cost);
    const auto cost12 = expr_names(p12, __cost);

    REQUIRE(contains_name(eq01, "eq_pred_owned_A"));
    REQUIRE(contains_name(eq01, "eq_pred_owned_B"));
    REQUIRE_FALSE(contains_name(eq01, "eq_pred_owned_C"));

    REQUIRE_FALSE(contains_name(eq12, "eq_pred_owned_A"));
    REQUIRE_FALSE(contains_name(eq12, "eq_pred_owned_B"));
    REQUIRE(contains_name(eq12, "eq_pred_owned_C"));

    REQUIRE(contains_name(cost01, "cost_pred_owned_A"));
    REQUIRE_FALSE(contains_name(cost01, "cost_pred_owned_B"));
    REQUIRE(contains_name(cost12, "cost_pred_owned_B"));
    REQUIRE(contains_name(cost12, "cost_pred_owned_C"));

    const auto &eqA = require_func_named(p01, __eq_x, "eq_pred_owned_A");
    const auto &eqB = require_func_named(p01, __eq_x, "eq_pred_owned_B");
    const auto &eqC = require_func_named(p12, __eq_x, "eq_pred_owned_C");
    const auto &costB = require_func_named(p12, __cost, "cost_pred_owned_B");
    const auto &costC = require_func_named(p12, __cost, "cost_pred_owned_C");

    REQUIRE(eqA.in_args().size() == 1);
    REQUIRE(eqB.in_args().size() == 1);
    REQUIRE(eqC.in_args().size() == 1);

    REQUIRE(eqA.in_args().front()->field() == __x);
    REQUIRE(eqB.in_args().front()->field() == __y);
    REQUIRE(eqC.in_args().front()->field() == __y);
    REQUIRE(costB.in_args().front()->field() == __x);
    REQUIRE(costC.in_args().front()->field() == __y);
}

TEST_CASE("graph_model compose materializes sink node state cost onto incoming edge y") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_sink", 1);
    auto u = sym::inputs("u_sink", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_sink", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto st_prob = node_ocp::create();
    auto st_cost = cost(new generic_cost("cost_A_sink", x_args, x, approx_order::second));
    st_prob->add(*st_cost);

    auto sink_prob = node_ocp::create();
    auto sink_cost = cost(new generic_cost("cost_B_sink", x_args, x, approx_order::second));
    sink_prob->add(*sink_cost);

    graph_model modeled;
    auto n0 = modeled.add_node(st_prob);
    auto n1 = modeled.add_node(sink_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 1);
    auto p01 = composed.front();

    const auto cost01 = expr_names(p01, __cost);
    REQUIRE(contains_name(cost01, "cost_A_sink"));
    REQUIRE(contains_name(cost01, "cost_B_sink"));

    const auto &costA = require_func_named(p01, __cost, "cost_A_sink");
    const auto &costB = require_func_named(p01, __cost, "cost_B_sink");

    REQUIRE(costA.in_args().size() == 1);
    REQUIRE(costB.in_args().size() == 1);
    REQUIRE(costA.in_args().front()->field() == __x);
    REQUIRE(costB.in_args().front()->field() == __y);
}

TEST_CASE("graph_model compose keeps explicit terminal sink cost on terminal node") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_term", 1);
    auto u = sym::inputs("u_term", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_term", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto st_prob = node_ocp::create();
    auto st_cost = cost(new generic_cost("cost_A_term", x_args, x, approx_order::second));
    st_prob->add(*st_cost);

    auto sink_prob = node_ocp::create();
    auto sink_cost = cost(new generic_cost("cost_B_term", x_args, x, approx_order::second));
    sink_prob->add_terminal(*sink_cost);

    graph_model modeled;
    auto n0 = modeled.add_node(st_prob);
    auto n1 = modeled.add_node(sink_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 1);
    auto p01 = composed.front();

    const auto cost01 = expr_names(p01, __cost);
    REQUIRE(contains_name(cost01, "cost_A_term"));
    REQUIRE_FALSE(contains_name(cost01, "cost_B_term"));

    auto p_terminal = modeled.compose_terminal(n1);
    const auto terminal_costs = expr_names(p_terminal, __cost);
    REQUIRE(contains_name(terminal_costs, "cost_B_term"));

    const auto &costA = require_func_named(p01, __cost, "cost_A_term");
    const auto &costB = require_func_named(p_terminal, __cost, "cost_B_term");
    REQUIRE(costA.in_args().size() == 1);
    REQUIRE(costB.in_args().size() == 1);
    REQUIRE(costA.in_args().front()->field() == __x);
    REQUIRE(costB.in_args().front()->field() == __x);
}

TEST_CASE("graph_model composed stages can be consumed by sqp create_node") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_solver", 1);
    auto u = sym::inputs("u_solver", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_solver_create_node_smoke", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto st_prob = node_ocp::create();
    st_prob->add(*constr(new generic_constr("eq_solver_create_node_smoke_a", x_args, x, approx_order::second, __undefined)));
    st_prob->add(*cost(new generic_cost("cost_solver_create_node_smoke_a", x_args, x, approx_order::second)));

    auto ed_prob = node_ocp::create();
    ed_prob->add(*constr(new generic_constr("eq_solver_create_node_smoke_b", x_args, x, approx_order::second, __undefined)));
    ed_prob->add(*cost(new generic_cost("cost_solver_create_node_smoke_b", x_args, x, approx_order::second)));

    graph_model modeled;
    auto n0 = modeled.add_node(st_prob);
    auto n1 = modeled.add_node(ed_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 1);

    ns_sqp sqp;
    auto node = sqp.create_node(composed.front());

    REQUIRE(node->problem().uid() == composed.front()->uid());
    REQUIRE(node->problem().dim(__x) == composed.front()->dim(__x));
    REQUIRE(node->problem().dim(__u) == composed.front()->dim(__u));
    REQUIRE(node->problem().dim(__y) == composed.front()->dim(__y));
}
