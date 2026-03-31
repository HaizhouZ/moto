#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <moto/model/graph_model.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

std::vector<std::string> expr_names(const moto::ocp_base_ptr_t &prob, moto::field_t field) {
    std::vector<std::string> names;
    for (const moto::shared_expr &expr : prob->exprs(field)) {
        names.push_back(expr->name());
    }
    return names;
}

std::vector<std::string> expr_names(const moto::ocp_base &prob, moto::field_t field) {
    std::vector<std::string> names;
    for (const moto::shared_expr &expr : prob.exprs(field)) {
        names.push_back(expr->name());
    }
    return names;
}

bool contains_name(const std::vector<std::string> &names, const std::string &target) {
    return std::find(names.begin(), names.end(), target) != names.end();
}

bool contains_name_prefix(const std::vector<std::string> &names, const std::string &prefix) {
    return std::any_of(names.begin(), names.end(), [&](const std::string &name) {
        return name.rfind(prefix, 0) == 0;
    });
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

const moto::generic_func &require_func_named_prefix(const moto::ocp_base_ptr_t &prob,
                                                    moto::field_t field,
                                                    const std::string &prefix) {
    auto it = std::find_if(prob->exprs(field).begin(), prob->exprs(field).end(), [&](const moto::shared_expr &expr) {
        return expr->name().rfind(prefix, 0) == 0;
    });
    REQUIRE(it != prob->exprs(field).end());
    const auto *func = dynamic_cast<const moto::generic_func *>((*it).get());
    REQUIRE(func != nullptr);
    return *func;
}
} // namespace

TEST_CASE("graph_model compose lowers node-owned state terms onto outgoing edge y") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x", 1);
    auto u = sym::inputs("u", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};
    const var_inarg_list u_args = var_list{u};

    auto dyn = dynamics(new dense_dynamics("dyn", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto node_prob = node_ocp::create();
    auto eq = constr(new generic_constr("eq_node", x_args, x, approx_order::second, __undefined));
    auto x_cost = cost(new generic_cost("cost_node", x_args, x, approx_order::second));
    auto u_cost = cost(new generic_cost("cost_u_node", u_args, u, approx_order::second));
    node_prob->add(*eq);
    node_prob->add(*x_cost);
    node_prob->add(*u_cost);

    graph_model modeled;
    auto n0 = modeled.add_node(node_prob);
    auto n1 = modeled.add_node();

    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 1);
    auto p01 = composed.front();

    const auto &eq_node = require_func_named_prefix(p01, __eq_x, "eq_node");
    const auto &cost_node = require_func_named_prefix(p01, __cost, "cost_node");
    const auto &cost_u_node = require_func_named(p01, __cost, "cost_u_node");

    REQUIRE(eq_node.in_args().front()->field() == __y);
    REQUIRE(cost_node.in_args().front()->field() == __y);
    REQUIRE(cost_u_node.in_args().front()->field() == __u);
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
    REQUIRE(contains_name_prefix(cost01, "cost_A_sink"));
    REQUIRE(contains_name_prefix(cost01, "cost_B_sink"));

    const auto &costA = require_func_named_prefix(p01, __cost, "cost_A_sink");
    const auto &costB = require_func_named_prefix(p01, __cost, "cost_B_sink");

    REQUIRE(costA.in_args().size() == 1);
    REQUIRE(costB.in_args().size() == 1);
    REQUIRE(costA.in_args().front()->field() == __y);
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
    auto st_cost = cost(new generic_cost("cost_graph_terminal_stage_unique", x_args, x, approx_order::second));
    st_prob->add(*st_cost);

    auto sink_prob = node_ocp::create();
    auto sink_cost = cost(new generic_cost("cost_graph_terminal_sink_unique", x_args, x, approx_order::second));
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
    REQUIRE(contains_name_prefix(cost01, "cost_graph_terminal_stage_unique"));
    REQUIRE_FALSE(contains_name(cost01, "cost_graph_terminal_sink_unique"));

    auto p_terminal = modeled.compose_terminal(n1);
    const auto terminal_costs = expr_names(p_terminal, __cost);
    REQUIRE(contains_name(terminal_costs, "cost_graph_terminal_sink_unique"));

    const auto &costA = require_func_named_prefix(p01, __cost, "cost_graph_terminal_stage_unique");
    const auto &costB = require_func_named(p_terminal, __cost, "cost_graph_terminal_sink_unique");
    REQUIRE(costA.in_args().size() == 1);
    REQUIRE(costB.in_args().size() == 1);
    REQUIRE(costA.in_args().front()->field() == __y);
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

    REQUIRE(node->problem().uid() != composed.front()->uid());
    REQUIRE(node->problem().dim(__x) == composed.front()->dim(__x));
    REQUIRE(node->problem().dim(__u) == composed.front()->dim(__u));
    REQUIRE(node->problem().dim(__y) == composed.front()->dim(__y));
}

TEST_CASE("ns_sqp create_node can compose graph_model edges directly") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_solver_direct", 1);
    auto u = sym::inputs("u_solver_direct", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_solver_direct", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto st_prob = node_ocp::create();
    st_prob->add(*cost(new generic_cost("cost_solver_direct_a", x_args, x, approx_order::second)));

    auto ed_prob = node_ocp::create();
    ed_prob->add(*cost(new generic_cost("cost_solver_direct_b", x_args, x, approx_order::second)));
    ed_prob->add_terminal(*cost(new generic_cost("cost_solver_direct_terminal", x_args, x, approx_order::second)));

    graph_model modeled;
    auto n0 = modeled.add_node(st_prob);
    auto n1 = modeled.add_node(ed_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    auto expected_edge = modeled.compose(e01);
    auto expected_terminal = modeled.compose_terminal(n1);

    ns_sqp sqp;
    auto edge_node = sqp.create_node(e01);
    auto terminal_node = sqp.create_terminal_node(n1);

    REQUIRE(edge_node->problem().uid() != expected_edge->uid());
    REQUIRE(terminal_node->problem().uid() != expected_terminal->uid());
    REQUIRE(edge_node->problem().dim(__x) == expected_edge->dim(__x));
    REQUIRE(edge_node->problem().dim(__u) == expected_edge->dim(__u));
    REQUIRE(edge_node->problem().dim(__y) == expected_edge->dim(__y));
    REQUIRE_FALSE(contains_name(expr_names(edge_node->problem(), __cost), "cost_solver_direct_terminal"));
    REQUIRE(contains_name(expr_names(terminal_node->problem(), __cost), "cost_solver_direct_terminal"));
}

TEST_CASE("ns_sqp create_terminal_node can materialize terminal sink cost from graph_model edge") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_solver_terminal_edge", 1);
    auto u = sym::inputs("u_solver_terminal_edge", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_solver_terminal_edge", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto st_prob = node_ocp::create();
    st_prob->add(*cost(new generic_cost("cost_solver_terminal_stage", x_args, x, approx_order::second)));

    auto terminal_cost = cost(new generic_cost("cost_solver_terminal_sink", x_args, x, approx_order::second));
    terminal_cost->as_terminal();
    auto ed_prob = node_ocp::create();
    ed_prob->add_terminal(*terminal_cost);

    graph_model modeled;
    auto n0 = modeled.add_node(st_prob);
    auto n1 = modeled.add_node(ed_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    ns_sqp sqp;
    auto terminal_node = sqp.create_terminal_node(e01);

    REQUIRE(contains_name_prefix(expr_names(terminal_node->problem(), __cost), "cost_solver_terminal_stage"));
    REQUIRE(contains_name_prefix(expr_names(terminal_node->problem(), __cost), "cost_solver_terminal_sink_terminal"));
}

TEST_CASE("ns_sqp create_node clones formulation templates") {
    using namespace moto;

    auto [x, xn] = sym::states("x_clone_template", 1);
    auto u = sym::inputs("u_clone_template", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto formulation = edge_ocp::create();
    formulation->add(*dynamics(new dense_dynamics("dyn_clone_template", dyn_args, xn - x - u, approx_order::second, __dyn)));
    formulation->add(*cost(new generic_cost("cost_clone_template", x_args, x, approx_order::second)));
    formulation->wait_until_ready();

    ns_sqp sqp;
    auto n0 = sqp.create_node(formulation);
    auto n1 = sqp.create_node(formulation);

    REQUIRE(n0->problem().uid() != formulation->uid());
    REQUIRE(n1->problem().uid() != formulation->uid());
    REQUIRE(n0->problem().uid() != n1->problem().uid());
    REQUIRE(n0->problem().dim(__x) == formulation->dim(__x));
    REQUIRE(n1->problem().dim(__y) == formulation->dim(__y));
}

TEST_CASE("ns_sqp create_nodes can batch clone formulation templates") {
    using namespace moto;

    auto [x, xn] = sym::states("x_batch_template", 1);
    auto u = sym::inputs("u_batch_template", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto formulation = edge_ocp::create();
    formulation->add(*dynamics(new dense_dynamics("dyn_batch_template", dyn_args, xn - x - u, approx_order::second, __dyn)));
    formulation->add(*cost(new generic_cost("cost_batch_template", x_args, x, approx_order::second)));
    formulation->wait_until_ready();

    ns_sqp sqp;
    std::vector<ocp::active_status_config> configs(3);
    auto nodes = sqp.create_nodes(formulation, configs);

    REQUIRE(nodes.size() == configs.size());
    REQUIRE(nodes[0]->problem().uid() != formulation->uid());
    REQUIRE(nodes[1]->problem().uid() != formulation->uid());
    REQUIRE(nodes[2]->problem().uid() != formulation->uid());
    REQUIRE(nodes[0]->problem().uid() != nodes[1]->problem().uid());
    REQUIRE(nodes[1]->problem().uid() != nodes[2]->problem().uid());
    REQUIRE(nodes[0]->problem().dim(__x) == formulation->dim(__x));
    REQUIRE(nodes[2]->problem().dim(__y) == formulation->dim(__y));
}

TEST_CASE("node_ocp rejects y-dependent terms and dynamics") {
    using namespace moto;

    auto [x, y] = sym::states("node_guard_x", 1);
    auto u = sym::inputs("node_guard_u", 1);
    const var_inarg_list x_args = var_list{x};
    const var_inarg_list y_args = var_list{y};
    const var_inarg_list dyn_args = var_list{x, y, u};

    auto node = node_ocp::create();

    auto x_cost = cost(new generic_cost("x_only_cost", x_args, x, approx_order::second));
    REQUIRE_NOTHROW(node->add(*x_cost));

    auto y_cost = cost(new generic_cost("y_only_cost", y_args, y, approx_order::second));
    REQUIRE_THROWS_WITH(
        node->add(*y_cost),
        Catch::Matchers::ContainsSubstring("node_ocp terms may only depend on x/u/p-style node variables"));

    auto dyn = dynamics(new dense_dynamics("node_guard_dyn", dyn_args, y - x - u, approx_order::second, __dyn));
    REQUIRE_THROWS_WITH(
        node->add(*dyn),
        Catch::Matchers::ContainsSubstring("dynamics must be added to an edge_ocp"));
}
