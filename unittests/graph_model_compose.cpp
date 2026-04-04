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
namespace cs = casadi;

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
    auto n0 = modeled.create_node(node_prob);
    auto n1 = modeled.create_node();

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
    auto n0 = modeled.create_node(st_prob);
    auto n1 = modeled.create_node(sink_prob);
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
    auto n0 = modeled.create_node(st_prob);
    auto n1 = modeled.create_node(sink_prob);
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
    auto n0 = modeled.create_node(st_prob);
    auto n1 = modeled.create_node(ed_prob);
    auto e01 = modeled.connect(n0, n1);
    e01->add(*dyn);

    const auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 1);

}

TEST_CASE("graph_model compose_interval applies projector-driven control and hard-constraint layout") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_projector_layout", 1);
    auto u0 = sym::inputs("u0_projector_layout", 1);
    auto u1 = sym::inputs("u1_projector_layout", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u0, u1};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_projector_layout",
                                           dyn_args,
                                           xn - x - u0 - cs::SX(2.0) * u1,
                                           approx_order::second, __dyn));

    auto node_prob = node_ocp::create();
    auto eq_state = constr(new generic_constr("eq_state_projector_layout",
                                              x_args,
                                              x,
                                              approx_order::second,
                                              __eq_x));
    node_prob->add(*eq_state);
    auto state_group = node_prob->projector().group();
    state_group.require_constraint(expr_list{eq_state});

    auto edge_prob = edge_ocp::create();
    auto eq_u0 = constr(new generic_constr("eq_u0_projector_layout",
                                           var_list{u0},
                                           u0,
                                           approx_order::second,
                                           __eq_xu));
    auto eq_u1 = constr(new generic_constr("eq_u1_projector_layout",
                                           var_list{u1},
                                           u1,
                                           approx_order::second,
                                           __eq_xu));
    edge_prob->add(*dyn);
    edge_prob->add(*eq_u0);
    edge_prob->add(*eq_u1);

    auto proj = edge_prob->projector();
    auto drive_group = proj.group();
    drive_group.require_primal(var_list{u1});
    drive_group.require_constraint(expr_list{eq_u1});
    auto balance_group = proj.group();
    balance_group.require_primal(var_list{u0});
    balance_group.require_constraint(expr_list{eq_u0});
    drive_group.require_before(state_group);
    state_group.require_before(balance_group);

    graph_model modeled;
    auto n0 = modeled.create_node(node_prob);
    auto n1 = modeled.create_node();
    auto e01 = modeled.connect(n0, n1, edge_prob);

    auto composed = modeled.compose_interval(e01);
    const std::vector<std::string> expected_u_names{
        "u1_projector_layout",
        "u0_projector_layout",
    };
    const std::vector<std::string> expected_eq_xu_names{
        "eq_u1_projector_layout",
        "eq_u0_projector_layout",
    };
    REQUIRE(expr_names(composed, __u) == expected_u_names);
    REQUIRE(expr_names(composed, __eq_xu) == expected_eq_xu_names);

    const auto &blocks = composed->compiled_hard_constraint_blocks();
    REQUIRE(blocks.size() == 3);
    REQUIRE(blocks[0].field == __eq_xu);
    REQUIRE(blocks[0].group_id == drive_group.id());
    REQUIRE(blocks[1].field == __eq_x);
    REQUIRE(blocks[1].group_id == state_group.id());
    REQUIRE(blocks[2].field == __eq_xu);
    REQUIRE(blocks[2].group_id == balance_group.id());

    ns_sqp::data stage(std::static_pointer_cast<ocp>(composed));
    REQUIRE(stage.default_elimination_stage_.rows.size() == 3);
    REQUIRE(stage.default_elimination_stage_.rows[0].kind == solver::projection::row_kind::state_input_eq);
    REQUIRE(stage.default_elimination_stage_.rows[1].kind == solver::projection::row_kind::projected_state_eq);
    REQUIRE(stage.default_elimination_stage_.rows[2].kind == solver::projection::row_kind::state_input_eq);
}

TEST_CASE("edge_ocp compose applies projector-driven layout without graph_model") {
    using namespace moto;

    auto [x, xn] = sym::states("x_direct_projector_layout", 1);
    auto u0 = sym::inputs("u0_direct_projector_layout", 1);
    auto u1 = sym::inputs("u1_direct_projector_layout", 1);

    auto node_prob = node_ocp::create();
    auto eq_state = constr(new generic_constr("eq_state_direct_projector_layout",
                                              var_list{x},
                                              x,
                                              approx_order::second,
                                              __eq_x));
    node_prob->add(*eq_state);
    auto state_group = node_prob->projector().group();
    state_group.require_constraint(expr_list{eq_state});

    auto edge_prob = edge_ocp::create();
    auto dyn = dynamics(new dense_dynamics("dyn_direct_projector_layout",
                                           var_list{x, xn, u0, u1},
                                           xn - x - u0 - cs::SX(2.0) * u1,
                                           approx_order::second,
                                           __dyn));
    auto eq_u0 = constr(new generic_constr("eq_u0_direct_projector_layout",
                                           var_list{u0},
                                           u0,
                                           approx_order::second,
                                           __eq_xu));
    auto eq_u1 = constr(new generic_constr("eq_u1_direct_projector_layout",
                                           var_list{u1},
                                           u1,
                                           approx_order::second,
                                           __eq_xu));
    edge_prob->add(*dyn);
    edge_prob->add(*eq_u0);
    edge_prob->add(*eq_u1);

    auto proj = edge_prob->projector();
    auto drive_group = proj.group();
    drive_group.require_primal(var_list{u1});
    drive_group.require_constraint(expr_list{eq_u1});
    auto balance_group = proj.group();
    balance_group.require_primal(var_list{u0});
    balance_group.require_constraint(expr_list{eq_u0});
    drive_group.require_before(state_group);
    state_group.require_before(balance_group);

    auto composed = edge_ocp::compose(node_prob, edge_prob);
    composed->wait_until_ready();

    REQUIRE(expr_names(composed, __u) == std::vector<std::string>{
                                          "u1_direct_projector_layout",
                                          "u0_direct_projector_layout",
                                      });
    REQUIRE(expr_names(composed, __eq_xu) == std::vector<std::string>{
                                             "eq_u1_direct_projector_layout",
                                             "eq_u0_direct_projector_layout",
                                         });

    const auto &blocks = composed->compiled_hard_constraint_blocks();
    REQUIRE(blocks.size() == 3);
    REQUIRE(blocks[0].group_id == drive_group.id());
    REQUIRE(blocks[1].group_id == state_group.id());
    REQUIRE(blocks[2].group_id == balance_group.id());
}

TEST_CASE("graph_model compose_interval rejects incompatible projector ownership") {
    using namespace moto;
    using namespace moto::model;
    using Catch::Matchers::ContainsSubstring;

    auto [x, xn] = sym::states("x_projector_conflict", 1);
    auto u = sym::inputs("u_projector_conflict", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};

    auto dyn = dynamics(new dense_dynamics("dyn_projector_conflict",
                                           dyn_args,
                                           xn - x - u,
                                           approx_order::second, __dyn));

    auto edge_prob = edge_ocp::create();
    edge_prob->add(*dyn);
    auto proj = edge_prob->projector();
    proj.group().require_primal(var_list{u});
    proj.group().require_primal(var_list{u});

    graph_model modeled;
    auto n0 = modeled.create_node();
    auto n1 = modeled.create_node();
    auto e01 = modeled.connect(n0, n1, edge_prob);

    REQUIRE_THROWS_WITH(modeled.compose_interval(e01), ContainsSubstring("projector layout conflict"));
}

TEST_CASE("ns_sqp create_graph can synchronize model graph paths into the internal directed graph") {
    using namespace moto;

    auto [x, xn] = sym::states("x_path_internal", 1);
    auto u = sym::inputs("u_path_internal", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};

    auto dyn = dynamics(new dense_dynamics("dyn_path_internal", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*cost(new generic_cost("cost_path_internal_stage", x_args, x, approx_order::second)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("cost_path_internal_terminal", x_args, x, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto stage_edges = modeled.add_path(n0, nt, 5);
    REQUIRE(stage_edges.size() == 5);
    for (const auto &edge : stage_edges) {
        edge->add(*dyn);
    }

    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == 5);
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_path_internal_terminal"));
}

TEST_CASE("ns_sqp model_graph add_path returns one edge per requested interval") {
    using namespace moto;

    auto [x, xn] = sym::states("x_path_topology", 1);
    auto u = moto::sym::inputs("u_path_topology", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};
    auto dyn = dynamics(new dense_dynamics("dyn_path_topology", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*cost(new generic_cost("cost_path_topology", x_args, x, approx_order::second)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("cost_path_topology_terminal", x_args, x, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto stage_edges = modeled.add_path(n0, nt, 3);
    REQUIRE(stage_edges.size() == 3);
    REQUIRE(modeled.num_edges() == 3);
    REQUIRE(modeled.num_nodes() == 4);
    for (const auto &edge : stage_edges) {
        edge->add(*dyn);
    }

    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == 3);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_path_topology"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_path_topology_terminal"));
}

TEST_CASE("ns_sqp model_graph add_path can create key nodes from node prototypes") {
    using namespace moto;

    auto [x, xn] = sym::states("x_path_proto", 1);
    auto u = moto::sym::inputs("u_path_proto", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};
    auto dyn = dynamics(new dense_dynamics("dyn_path_proto", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*cost(new generic_cost("cost_path_proto", x_args, x, approx_order::second)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("cost_path_proto_terminal", x_args, x, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto stage_edges = modeled.add_path(n0, nt, 2);
    REQUIRE(stage_edges.size() == 2);
    REQUIRE(modeled.num_edges() == 2);
    REQUIRE(modeled.num_nodes() == 3);
    for (const auto &edge : stage_edges) {
        edge->add(*dyn);
    }

    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == 2);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_path_proto"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_path_proto_terminal"));
}

TEST_CASE("ns_sqp terminal u-dependent terms are ignored instead of lowered onto the final edge") {
    using namespace moto;

    auto [x, xn] = sym::states("x_terminal_u_guard", 1);
    auto u = moto::sym::inputs("u_terminal_u_guard", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list xu_args = var_list{x, u};
    const var_inarg_list x_args = var_list{x};
    auto dyn = dynamics(new dense_dynamics("dyn_terminal_u_guard", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*cost(new generic_cost("cost_terminal_u_guard_stage", x_args, x, approx_order::second)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("cost_terminal_u_guard_terminal_x", x_args, x, approx_order::second)));
    terminal_prob->add_terminal(*cost(new generic_cost("cost_terminal_u_guard_terminal_xu", xu_args, x + u, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto edges = modeled.add_path(n0, nt, 2);
    for (const auto &edge : edges) {
        edge->add(*dyn);
    }

    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == 2);
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_terminal_u_guard_terminal_x"));
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_terminal_u_guard_terminal_xu"));
}

TEST_CASE("ns_sqp model_graph flatten_nodes reuses realized graph until graph becomes dirty") {
    using namespace moto;

    auto [x, xn] = sym::states("x_flatten_cache", 1);
    auto u = sym::inputs("u_flatten_cache", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};
    auto dyn = dynamics(new dense_dynamics("dyn_flatten_cache", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto stage_prob = node_ocp::create();
    stage_prob->add(*cost(new generic_cost("cost_flatten_cache_stage", x_args, x, approx_order::second)));

    auto terminal_prob = node_ocp::create();
    terminal_prob->add_terminal(*cost(new generic_cost("cost_flatten_cache_terminal", x_args, x, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    auto n0 = modeled.create_node(stage_prob);
    auto nt = modeled.create_node(terminal_prob);
    auto stage_edges = modeled.add_path(n0, nt, 2);
    for (const auto &edge : stage_edges) {
        edge->add(*dyn);
    }

    auto &flat_first = sqp.graph().flatten_nodes();
    REQUIRE(flat_first.size() == 2);
    const auto first_head_addr = static_cast<const void *>(flat_first.front());
    const auto first_tail_addr = static_cast<const void *>(flat_first.back());

    auto &flat_second = sqp.graph().flatten_nodes();
    REQUIRE(flat_second.size() == flat_first.size());
    REQUIRE(static_cast<const void *>(flat_second.front()) == first_head_addr);
    REQUIRE(static_cast<const void *>(flat_second.back()) == first_tail_addr);

    auto extra_terminal = modeled.create_node(terminal_prob->clone_node());
    auto extra_edges = modeled.add_path(nt, extra_terminal, 1);
    REQUIRE(extra_edges.size() == 1);
    extra_edges.front()->add(*dyn);

    auto &flat_after_dirty = sqp.graph().flatten_nodes();
    REQUIRE(flat_after_dirty.size() == 3);
    REQUIRE(static_cast<const void *>(flat_after_dirty.front()) != nullptr);
}

TEST_CASE("graph_model reserve supports bulk node and edge creation") {
    using namespace moto;
    using namespace moto::model;

    auto [x, xn] = sym::states("x_graph_reserve", 1);
    auto u = sym::inputs("u_graph_reserve", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    auto dyn = dynamics(new dense_dynamics("dyn_graph_reserve", dyn_args, xn - x - u, approx_order::second, __dyn));

    graph_model modeled;
    modeled.reserve(8, 8);

    auto n0 = modeled.create_node(node_ocp::create());
    auto nt = modeled.create_node(node_ocp::create());
    auto edges = modeled.add_path(n0, nt, 5);
    REQUIRE(edges.size() == 5);
    REQUIRE(modeled.num_nodes() == 6);
    REQUIRE(modeled.num_edges() == 5);
    for (const auto &edge : edges) {
        edge->add(*dyn);
    }

    auto composed = modeled.compose_all();
    REQUIRE(composed.size() == 5);
}

TEST_CASE("ns_sqp create_graph realizes a reserved multi-segment path topology") {
    using namespace moto;

    auto [x, xn] = sym::states("x_reserved_realize", 1);
    auto u = sym::inputs("u_reserved_realize", 1);
    const var_inarg_list dyn_args = var_list{x, xn, u};
    const var_inarg_list x_args = var_list{x};
    auto dyn = dynamics(new dense_dynamics("dyn_reserved_realize", dyn_args, xn - x - u, approx_order::second, __dyn));

    auto source_prob = node_ocp::create();
    source_prob->add(*cost(new generic_cost("cost_reserved_source", x_args, x, approx_order::second)));

    auto mid_prob = node_ocp::create();
    mid_prob->add(*cost(new generic_cost("cost_reserved_mid", x_args, x, approx_order::second)));

    auto sink_prob = node_ocp::create();
    sink_prob->add_terminal(*cost(new generic_cost("cost_reserved_sink", x_args, x, approx_order::second)));

    ns_sqp sqp;
    auto modeled = sqp.create_graph();
    modeled.reserve(4, 4);

    auto src = modeled.create_node(source_prob);
    auto mid = modeled.create_node(mid_prob);
    auto sink = modeled.create_node(sink_prob);

    auto e_src_mid = modeled.connect(src, mid);
    auto e_mid_sink = modeled.connect(mid, sink);
    for (const auto &edge : {e_src_mid, e_mid_sink}) {
        edge->add(*dyn);
    }

    auto &flat = sqp.graph().flatten_nodes();
    REQUIRE(flat.size() == 2);
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
