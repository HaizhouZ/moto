#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/graph_model.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

std::vector<std::string> expr_names(const moto::ocp_base &prob, moto::field_t field) {
    std::vector<std::string> names;
    for (const moto::shared_expr &expr : prob.exprs(field)) {
        names.push_back(expr->name());
    }
    return names;
}

bool contains_name_prefix(const std::vector<std::string> &names, const std::string &prefix) {
    return std::any_of(names.begin(), names.end(), [&](const std::string &name) {
        return name.rfind(prefix, 0) == 0;
    });
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

moto::node_ocp_ptr_t make_stage(const std::string &tag,
                                const moto::sym &x,
                                const moto::sym &u) {
    using namespace moto;
    auto stage = node_ocp::create();
    stage->add(*constr(new generic_constr("ineq_" + tag, var_list{x}, x, approx_order::second, __ineq_x)));
    stage->add(*cost(new generic_cost("cost_x_" + tag, var_list{x}, x * x, approx_order::second)));
    stage->add(*cost(new generic_cost("cost_u_" + tag, var_list{u}, u * u, approx_order::second)));
    return stage;
}

moto::edge_ocp_ptr_t make_edge(const std::string &tag,
                               const moto::sym &x,
                               const moto::sym &xn,
                               const moto::sym &u) {
    using namespace moto;
    auto edge = edge_ocp::create();
    edge->add(*dynamics(new dense_dynamics("dyn_" + tag, var_list{x, xn, u}, xn - x - u, approx_order::second, __dyn)));
    return edge;
}

std::vector<moto::ocp_ptr_t> realized_stages(moto::ns_sqp &sqp) {
    const auto &flat = sqp.solver_nodes();
    std::vector<moto::ocp_ptr_t> stages;
    stages.reserve(flat.size());
    for (const auto *stage : flat) {
        stages.push_back(stage->problem_ptr());
    }
    return stages;
}
} // namespace

TEST_CASE("graph_model add_path builds node-stage intervals and lowers pure state terms onto y") {
    using namespace moto;

    auto [x, xn] = sym::states("x_edge_stage", 1);
    auto u = sym::inputs("u_edge_stage", 1);
    auto stage = make_stage("node_stage", x, u);
    auto edge = make_edge("node_stage", x, xn, u);

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    modeled.add_path(stage, stage, edge, 3);

    const auto stages = realized_stages(sqp);
    REQUIRE(stages.size() == 3);
    const auto &ineq = require_func_named_prefix(stages.front(), __ineq_x, "ineq_node_stage");
    const auto &cost_x = require_func_named_prefix(stages.front(), __cost, "cost_x_node_stage");
    const auto &cost_u = require_func_named_prefix(stages.front(), __cost, "cost_u_node_stage");
    REQUIRE(ineq.in_args().front()->field() == __y);
    REQUIRE(cost_x.in_args().front()->field() == __y);
    REQUIRE(cost_u.in_args().front()->field() == __u);
}

TEST_CASE("graph_model reuses cached lowered function entities across stages") {
    using namespace moto;

    auto [x, xn] = sym::states("x_reuse_lowered", 1);
    auto u = sym::inputs("u_reuse_lowered", 1);
    auto stage = make_stage("reuse_lowered", x, u);
    auto edge = make_edge("reuse_lowered", x, xn, u);

    ns_sqp sqp;
    sqp.graph().add_path(stage, stage, edge, 4);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 4);

    const generic_func *first_lowered = nullptr;
    for (const auto *node : flat) {
        const auto &lowered = require_func_named_prefix(node->problem_ptr(), __cost, "cost_x_reuse_lowered");
        REQUIRE(lowered.in_args().front()->field() == __y);
        if (first_lowered == nullptr) {
            first_lowered = &lowered;
        } else {
            REQUIRE(&lowered == first_lowered);
            REQUIRE(lowered.uid() == first_lowered->uid());
        }
    }
}

TEST_CASE("graph_model realized stages are stable under concurrent readers") {
    using namespace moto;

    auto [x, xn] = sym::states("x_concurrent_realize", 1);
    auto u = sym::inputs("u_concurrent_realize", 1);
    auto stage = make_stage("concurrent_realize", x, u);
    auto edge = make_edge("concurrent_realize", x, xn, u);

    ns_sqp sqp;
    sqp.graph().add_path(stage, stage, edge, 4);

    std::atomic<size_t> observed{0};
    std::vector<std::thread> threads;
    for (size_t tid = 0; tid < 8; ++tid) {
        threads.emplace_back([&]() {
            for (size_t iter = 0; iter < 20; ++iter) {
                observed.fetch_add(sqp.solver_nodes().size());
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    REQUIRE(observed.load() == 8 * 20 * 4);
}

TEST_CASE("remap cache reuses keyed concrete function clones") {
    using namespace moto;

    auto [x_a, y_a] = sym::states("x_remap_cache_a", 1);
    auto y_b = sym::states("x_remap_cache_b", 1).second;
    auto p = sym::params("p_remap_cache", 1);
    auto u_remap = sym::inputs("u_remap_cache", 1);
    auto c = cost(new generic_cost("cost_remap_cache", var_list{x_a, p}, x_a * x_a + p, approx_order::second));
    REQUIRE(c->finalize(true));

    auto first = c->remap_arguments({{x_a, y_a}});
    auto second = c->remap_arguments({{x_a, y_a}});
    auto duplicate = c->remap_arguments({{x_a, y_a}, {x_a, y_a}});
    auto identity = c->remap_arguments({{x_a, x_a}});
    auto different = c->remap_arguments({{x_a, y_b}});

    REQUIRE(first.get() == second.get());
    REQUIRE(first.get() == duplicate.get());
    REQUIRE(identity.get() == c.get());
    REQUIRE(first.get() != different.get());
    REQUIRE(first.as<generic_func>().in_args().front()->uid() == y_a->uid());
    REQUIRE(first.as<generic_func>().arg_num(__x) == 0);
    REQUIRE(first.as<generic_func>().arg_num(__y) == 1);
    REQUIRE(different.as<generic_func>().in_args().front()->uid() == y_b->uid());
    REQUIRE_THROWS_AS(c->remap_arguments({{x_a, p}}), std::runtime_error);
    REQUIRE_THROWS_AS(c->remap_arguments({{p, u_remap}}), std::runtime_error);
    auto y_bad_dim = sym::states("x_remap_bad_dim", 2).second;
    auto p_bad_dim = sym::params("p_remap_bad_dim", 2);
    REQUIRE_THROWS_AS(c->remap_arguments({{x_a, y_bad_dim}}), std::runtime_error);
    REQUIRE_THROWS_AS(c->remap_arguments({{p, p_bad_dim}}), std::runtime_error);

    std::vector<shared_expr> threaded_results(16);
    std::vector<std::thread> remap_threads;
    for (size_t i = 0; i < threaded_results.size(); ++i) {
        remap_threads.emplace_back([&, i]() {
            threaded_results[i] = c->remap_arguments({{x_a, y_a}});
        });
    }
    for (auto &thread : remap_threads) {
        thread.join();
    }
    for (const auto &result : threaded_results) {
        REQUIRE(result.get() == first.get());
    }

    auto [x, y] = sym::states("x_concrete_clone", 1);
    auto u = sym::inputs("u_concrete_clone", 1);
    auto dyn = std::make_shared<dense_dynamics>(
        "dyn_concrete_clone", var_list{x, y, u}, y - x - u, approx_order::second, __dyn);
    auto ineq = ineq_constr::create(
        "ineq_concrete_clone", var_list{x}, x, approx_order::first, __ineq_x);

    REQUIRE(dynamic_cast<dense_dynamics *>(shared_expr(dyn).clone().get()) != nullptr);

    REQUIRE(ineq->finalize(true));
    auto remapped = ineq->remap_arguments({{x, y}});
    REQUIRE(dynamic_cast<ineq_constr *>(remapped.get()) != nullptr);
    REQUIRE(remapped.as<generic_func>().in_args().front()->uid() == y->uid());
}

TEST_CASE("generic_func share applies explicit argument remaps") {
    using namespace moto;

    auto [x, _] = sym::states("x_share_layout", 1);
    auto p = sym::params("p_share_layout", 1);
    auto c = cost(new generic_cost("cost_share_layout", var_list{x, p}, x * x + p, approx_order::second));
    REQUIRE(c->finalize(true));

    auto x_copy = x->clone("x_share_layout_copy");
    auto p_copy = p->clone("p_share_layout_copy");
    auto shared = c->share(generic_func::symbol_remap{{x, x_copy}, {p, p_copy}});
    REQUIRE(shared.finalized());
    REQUIRE_FALSE(shared.has_arg(x));
    REQUIRE_FALSE(shared.has_arg(p));
    REQUIRE(shared.has_arg(x_copy));
    REQUIRE(shared.has_arg(p_copy));
    REQUIRE(shared.in_args().size() == 2);
    REQUIRE(shared.arg_idx(shared.in_args(0)) == 0);
    REQUIRE(shared.arg_idx(shared.in_args(1)) == 1);
    REQUIRE(shared.in_args(__x).front()->uid() == x_copy->uid());
    REQUIRE(shared.in_args(__p).front()->uid() == p_copy->uid());

    auto x_partial = x->clone("x_share_layout_partial");
    auto partial = c->share(generic_func::symbol_remap{{x, x_partial}});
    REQUIRE_FALSE(partial.has_arg(x));
    REQUIRE(partial.has_arg(x_partial));
    REQUIRE(partial.has_arg(p));
    REQUIRE(partial.in_args(__x).front()->uid() == x_partial->uid());
    REQUIRE(partial.in_args(__p).front()->uid() == p->uid());

    auto same_args = c->share();
    REQUIRE(same_args.has_arg(x));
    REQUIRE(same_args.has_arg(p));
    REQUIRE(same_args.in_args(__x).front()->uid() == x->uid());
    REQUIRE(same_args.in_args(__p).front()->uid() == p->uid());

    auto y_bad_dim = sym::states("x_share_bad_dim", 2).second;
    auto p_bad_dim = sym::params("p_share_bad_dim", 2);
    REQUIRE_THROWS_AS(c->share(generic_func::symbol_remap{{x, y_bad_dim}}), std::runtime_error);
    REQUIRE_THROWS_AS(c->share(generic_func::symbol_remap{{p, p_bad_dim}}), std::runtime_error);
}

TEST_CASE("generic_func share is source-read-only for codegen and callback funcs") {
    using namespace moto;

    auto [x_codegen, _] = sym::states("x_share_codegen_thread", 1);
    auto p_codegen = sym::params("p_share_codegen_thread", 1);
    auto codegen = cost(new generic_cost("cost_share_codegen_thread",
                                         var_list{x_codegen, p_codegen},
                                         x_codegen * x_codegen + p_codegen,
                                         approx_order::second));
    REQUIRE(codegen->finalize(true));

    auto [x_callback, __] = sym::states("x_share_callback_thread", 1);
    auto p_callback = sym::params("p_share_callback_thread", 1);
    auto callback = cost(new generic_cost("cost_share_callback_thread", approx_order::first));
    callback->add_arguments({x_callback, p_callback});
    callback->value = [](func_approx_data &d) { d.v_.setZero(); };
    callback->jacobian = [](func_approx_data &d) {
        for (auto &jac : d.jac_) {
            if (jac.size() > 0) {
                jac.setZero();
            }
        }
    };
    REQUIRE(callback->finalize(true));

    const auto check_concurrent_share = [](const generic_func &func, const var &original_arg) {
        std::atomic<int> failures{0};
        std::vector<std::thread> threads;
        for (size_t tid = 0; tid < 8; ++tid) {
            threads.emplace_back([&]() {
                for (size_t iter = 0; iter < 20; ++iter) {
                    auto replacement = original_arg->clone(original_arg->name() + "_shared");
                    auto shared = func.share(generic_func::symbol_remap{{original_arg, replacement}});
                    if (!shared.finalized() || !shared.value || !shared.jacobian) {
                        ++failures;
                    }
                    if (shared.in_args().empty() || shared.has_arg(original_arg) || !shared.has_arg(replacement)) {
                        ++failures;
                    }
                    if (shared.in_args(original_arg->field()).empty()) {
                        ++failures;
                    }
                }
            });
        }
        for (auto &thread : threads) {
            thread.join();
        }
        REQUIRE(failures.load() == 0);
    };

    check_concurrent_share(*codegen, x_codegen);
    check_concurrent_share(*callback, x_callback);
}

TEST_CASE("graph_model add_path appends node-stage segments") {
    using namespace moto;

    auto [x, xn] = sym::states("x_append_stage", 1);
    auto u = sym::inputs("u_append_stage", 1);
    auto stage_a = make_stage("append_a", x, u);
    auto stage_b = make_stage("append_b", x, u);
    auto edge = make_edge("append", x, xn, u);

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    modeled.add_path(stage_a, stage_b, edge, 1);

    REQUIRE(sqp.solver_nodes().size() == 1);

    modeled.add_path(stage_b, stage_b, edge, 2);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_u_append_a"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_u_append_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_x_append_b"));
}

TEST_CASE("graph_model materializes terminal sink state terms on final edge only") {
    using namespace moto;

    auto [x, xn] = sym::states("x_terminal_sink", 1);
    auto u = sym::inputs("u_terminal_sink", 1);
    auto stage = make_stage("terminal_sink", x, u);
    auto edge = make_edge("terminal_sink", x, xn, u);

    auto terminal = node_ocp::create();
    terminal->add_terminal(*cost(new generic_cost("cost_terminal_sink_x", var_list{x}, x * x, approx_order::second)));
    terminal->add_terminal(*cost(new generic_cost("cost_terminal_sink_xu", var_list{x, u}, x + u, approx_order::second)));

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    modeled.add_path(stage, terminal, edge, 2);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 2);
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_terminal_sink_x"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_terminal_sink_x"));
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_terminal_sink_xu"));

    const auto &terminal_cost = require_func_named_prefix(flat.back()->problem_ptr(), __cost, "cost_terminal_sink_x");
    REQUIRE(terminal_cost.in_args().front()->field() == __y);
}

TEST_CASE("graph_model does not lower edge-local pure state terms") {
    using namespace moto;

    auto [x, xn] = sym::states("x_edge_local", 1);
    auto u = sym::inputs("u_edge_local", 1);
    auto stage = node_ocp::create();
    auto edge = make_edge("edge_local", x, xn, u);
    edge->add(*cost(new generic_cost("cost_edge_local_x", var_list{x}, x * x, approx_order::second)));

    ns_sqp sqp;
    sqp.graph().add_path(stage, stage, edge, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 1);
    const auto &edge_cost = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_edge_local_x");
    REQUIRE(edge_cost.in_args().front()->field() == __x);
}

TEST_CASE("graph_model honors node-stage active status during composition") {
    using namespace moto;

    auto [x, xn] = sym::states("x_edge_active", 1);
    auto ua = sym::inputs("u_edge_keep", 1);
    auto ub = sym::inputs("u_edge_drop", 1);

    auto stage = node_ocp::create();
    stage->add(*cost(new generic_cost("cost_edge_active_u", var_list{ua, ub}, ua * ua + ub * ub, approx_order::second)));
    auto edge = edge_ocp::create();
    edge->add(*dynamics(new dense_dynamics("dyn_edge_active", var_list{x, xn, ua, ub}, xn - x - ua - ub, approx_order::second, __dyn)));

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    auto active_stage = stage->clone_node(ocp::active_status_config{{ub}, {}});
    modeled.add_path(active_stage, active_stage, edge, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 1);
    REQUIRE(flat.front()->problem().dim(__u) == 1);
    REQUIRE(expr_names(flat.front()->problem(), __u) == std::vector<std::string>{"u_edge_keep"});
}

TEST_CASE("graph_model snapshots edge prototype when adding a path") {
    using namespace moto;

    auto [x, xn] = sym::states("x_formulation_dirty", 1);
    auto u = sym::inputs("u_formulation_dirty", 1);
    auto stage = make_stage("formulation_dirty", x, u);
    auto edge_stage = make_edge("formulation_dirty", x, xn, u);

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    modeled.add_path(stage, stage, edge_stage, 1);

    auto &flat_first = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_first.front()->problem(), __cost), "cost_added_after_realize"));

    edge_stage->add(*cost(new generic_cost("cost_added_after_realize", var_list{x}, x, approx_order::second)));

    auto &flat_after_mutation = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_after_mutation.front()->problem(), __cost), "cost_added_after_realize"));
}

TEST_CASE("graph_model snapshots endpoint prototype when adding a path") {
    using namespace moto;

    auto [x, xn] = sym::states("x_endpoint_dirty", 1);
    auto u = sym::inputs("u_endpoint_dirty", 1);
    auto stage = make_stage("endpoint_dirty", x, u);
    auto edge_stage = make_edge("endpoint_dirty", x, xn, u);

    ns_sqp sqp;
    auto &modeled = sqp.graph();
    modeled.add_path(stage, stage, edge_stage, 1);

    auto &flat_first = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_first.front()->problem(), __cost), "cost_endpoint_added_after_realize"));

    stage->add(*cost(new generic_cost("cost_endpoint_added_after_realize", var_list{x}, x, approx_order::second)));

    auto &flat_after_mutation = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_after_mutation.front()->problem(), __cost), "cost_endpoint_added_after_realize"));
}

TEST_CASE("node_ocp rejects y-dependent terms and dynamics") {
    using namespace moto;

    auto [x, y] = sym::states("node_guard_x", 1);
    auto u = sym::inputs("node_guard_u", 1);
    auto node = node_ocp::create();

    REQUIRE_NOTHROW(node->add(*cost(new generic_cost("x_only_cost", var_list{x}, x, approx_order::second))));
    REQUIRE_THROWS_WITH(
        node->add(*cost(new generic_cost("y_only_cost", var_list{y}, y, approx_order::second))),
        Catch::Matchers::ContainsSubstring("node_ocp terms may only depend on x/u/p-style node variables"));
    REQUIRE_THROWS_WITH(
        node->add(*dynamics(new dense_dynamics("node_guard_dyn", var_list{x, y, u}, y - x - u, approx_order::second, __dyn))),
        Catch::Matchers::ContainsSubstring("dynamics must be added to an edge_ocp"));
}

TEST_CASE("ocp active status can reactivate disabled expressions") {
    using namespace moto;

    auto x = sym::states("x_active_reactivate", 1).first;
    auto x_cost = cost(new generic_cost("cost_active_reactivate", var_list{x}, x, approx_order::second));
    auto node = node_ocp::create();
    node->add(*x_cost);

    node->update_active_status({{*x_cost}, {}});
    REQUIRE_FALSE(node->is_active(*x_cost));
    REQUIRE_NOTHROW(node->update_active_status({{}, {*x_cost}}));
    REQUIRE(node->is_active(*x_cost));
}
