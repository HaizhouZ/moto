#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
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

template <typename FuncPtr>
FuncPtr add_args(FuncPtr f, const moto::var_inarg_list &args) {
    f->add_arguments(args);
    return f;
}

moto::dynamics layout_dynamics(const std::string &name,
                               const moto::var_inarg_list &args,
                               size_t dim = 1) {
    return add_args(moto::dynamics(new moto::dense_dynamics(
                        name, moto::approx_order::second, dim, moto::__dyn)),
                    args);
}

moto::constr layout_constr(const std::string &name,
                           const moto::var_inarg_list &args,
                           moto::field_t field,
                           size_t dim = 1) {
    return add_args(moto::constr(new moto::generic_constr(
                        name, moto::approx_order::second, dim, field)),
                    args);
}

moto::cost layout_cost(const std::string &name,
                       const moto::var_inarg_list &args) {
    return add_args(moto::cost(new moto::generic_cost(
                        name, moto::approx_order::second)),
                    args);
}

size_t expr_dim(const moto::var &v) {
    return v.as<moto::expr>().dim();
}

size_t expr_dim(const moto::sym &s) {
    return static_cast<const moto::expr &>(s).dim();
}

moto::dynamics callback_linear_dynamics(const std::string &name,
                                        const moto::var &x,
                                        const moto::var &y,
                                        const moto::var &u) {
    using namespace moto;
    auto dyn = layout_dynamics(name, var_list{x, y, u}, expr_dim(y));
    dyn->value = [](func_approx_data &d) {
        d.v_ = d[1] - d[0] - d[2];
    };
    dyn->jacobian = [](func_approx_data &d) {
        d.jac_[0].setZero();
        d.jac_[1].setZero();
        d.jac_[2].setZero();
        d.jac_[0].diagonal().array() = -1.;
        d.jac_[1].diagonal().array() = 1.;
        d.jac_[2].diagonal().array() = -1.;
    };
    dyn->hessian = [](func_approx_data &) {};
    return dyn;
}

moto::cost callback_quadratic_cost(const std::string &name,
                                   const moto::var &x,
                                   moto::scalar_t target = 0.) {
    using namespace moto;
    auto c = layout_cost(name, var_list{x});
    c->value = [target](func_approx_data &d) {
        const scalar_t r = d[0](0) - target;
        d.v_(0) += r * r;
    };
    c->jacobian = [target](func_approx_data &d) {
        d.jac_[0](0, 0) += scalar_t(2.) * (d[0](0) - target);
    };
    c->hessian = [](func_approx_data &d) {
        d.lag_hess_[0][0](0, 0) += scalar_t(2.);
    };
    return c;
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

moto::stage_ocp_ptr_t make_stage(const std::string &tag,
                                 const moto::sym &x,
                                 const moto::sym &y,
                                 const moto::sym &u) {
    using namespace moto;
    auto stage = stage_ocp::create();
    stage->add(*layout_dynamics("dyn_" + tag, var_list{x, y, u}, expr_dim(y)));
    stage->add(*layout_constr("ineq_" + tag, var_list{x}, __ineq_x, expr_dim(x)));
    stage->add(*layout_cost("cost_x_" + tag, var_list{x}));
    stage->add(*layout_cost("cost_u_" + tag, var_list{u}));
    return stage;
}

} // namespace

TEST_CASE("stage graph maps stage, start-node, and end-node terms to solver fields", "[graph][mapping]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_edge_stage", 1);
    auto u = sym::inputs("u_edge_stage", 1);
    auto stage = make_stage("node_stage", x, xn, u);
    stage->st().add(*layout_cost("cost_st_node_stage", var_list{x}));
    stage->ed().add(*layout_cost("cost_ed_node_stage", var_list{x}));

    ns_sqp sqp;
    sqp.add_stage(stage, 3);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    const auto &ineq = require_func_named_prefix(flat.front()->problem_ptr(), __ineq_x, "ineq_node_stage");
    const auto &cost_x = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_x_node_stage");
    const auto &cost_st = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_st_node_stage");
    const auto &cost_ed = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_ed_node_stage");
    const auto &cost_u = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_u_node_stage");
    REQUIRE(ineq.in_args().front()->field() == __x);
    REQUIRE(cost_x.in_args().front()->field() == __x);
    REQUIRE(cost_st.in_args().front()->field() == __y);
    REQUIRE(cost_ed.in_args().front()->field() == __y);
    REQUIRE(cost_u.in_args().front()->field() == __u);
}

TEST_CASE("graph start terms are explicit while stage starts lower through incoming boundaries", "[graph][mapping]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_graph_start_rule", 1);
    auto u = sym::inputs("u_graph_start_rule", 1);
    auto stage = make_stage("graph_start_rule", x, xn, u);
    stage->st().add(*layout_cost("cost_stage_start_graph_start_rule", var_list{x}));

    ns_sqp sqp;
    sqp.start_node().add(*layout_cost("cost_graph_start_rule", var_list{x}));
    sqp.add_stage(stage, 2);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 2);

    const auto &initial = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_graph_start_rule");
    const auto &boundary = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_stage_start_graph_start_rule");
    REQUIRE(initial.in_args().front()->field() == __x);
    REQUIRE(boundary.in_args().front()->field() == __y);
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_stage_start_graph_start_rule"));
}

TEST_CASE("graph_model reuses cached lowered function entities across stages", "[graph][remap]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_reuse_lowered", 1);
    auto u = sym::inputs("u_reuse_lowered", 1);
    auto stage = make_stage("reuse_lowered", x, xn, u);
    stage->ed().add(*layout_cost("cost_ed_reuse_lowered", var_list{x}));

    ns_sqp sqp;
    sqp.add_stage(stage, 4);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 4);

    const generic_func *first_lowered = nullptr;
    for (const auto *node : flat) {
        const auto &lowered = require_func_named_prefix(node->problem_ptr(), __cost, "cost_ed_reuse_lowered");
        REQUIRE(lowered.in_args().front()->field() == __y);
        if (first_lowered == nullptr) {
            first_lowered = &lowered;
        } else {
            REQUIRE(&lowered == first_lowered);
            REQUIRE(lowered.uid() == first_lowered->uid());
        }
    }
}

TEST_CASE("graph_model realized stages are stable under concurrent readers", "[graph][concurrency]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_concurrent_realize", 1);
    auto u = sym::inputs("u_concurrent_realize", 1);
    auto stage = make_stage("concurrent_realize", x, xn, u);

    ns_sqp sqp;
    sqp.add_stage(stage, 4);

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

TEST_CASE("remap cache reuses keyed concrete function clones", "[graph][remap]") {
    using namespace moto;

    auto [x_a, y_a] = sym::states("x_remap_cache_a", 1);
    auto y_b = sym::states("x_remap_cache_b", 1).second;
    auto p = sym::params("p_remap_cache", 1);
    auto u_remap = sym::inputs("u_remap_cache", 1);
    auto c = layout_cost("cost_remap_cache", var_list{x_a, p});
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

TEST_CASE("sqp add_stage appends repeated stage segments from the graph tail", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_append_stage", 1);
    auto u = sym::inputs("u_append_stage", 1);
    auto stage_a = make_stage("append_a", x, xn, u);
    auto stage_b = make_stage("append_b", x, xn, u);

    ns_sqp sqp;
    auto first = sqp.add_stage(stage_a, 1);

    REQUIRE(sqp.solver_nodes().size() == 1);

    auto second = sqp.add_stage(stage_b, 2);
    REQUIRE(first.size() == 1);
    REQUIRE(second.size() == 2);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_u_append_a"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_u_append_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_x_append_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_x_append_b"));
    const auto &stage_cost = require_func_named_prefix(flat.back()->problem_ptr(), __cost, "cost_x_append_b");
    REQUIRE(stage_cost.in_args().front()->field() == __x);
}

TEST_CASE("sqp add_stage lowers the next phase start endpoint onto the previous tail", "[graph][mapping]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_phase_boundary", 1);
    auto u = sym::inputs("u_phase_boundary", 1);
    auto stage_a = make_stage("phase_boundary_a", x, xn, u);
    auto stage_b = make_stage("phase_boundary_b", x, xn, u);
    stage_a->ed().add(*layout_cost("cost_ed_phase_boundary_a", var_list{x}));
    stage_b->st().add(*layout_cost("cost_st_phase_boundary_b", var_list{x}));
    stage_b->ed().add(*layout_cost("cost_ed_phase_boundary_b", var_list{x}));

    ns_sqp sqp;
    sqp.add_stage(stage_a, 2);
    sqp.add_stage(stage_b, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);

    const auto first_names = expr_names(flat.front()->problem(), __cost);
    const auto boundary_names = expr_names(flat.at(1)->problem(), __cost);
    const auto tail_names = expr_names(flat.back()->problem(), __cost);

    REQUIRE(contains_name_prefix(first_names, "cost_ed_phase_boundary_a"));
    REQUIRE(contains_name_prefix(boundary_names, "cost_ed_phase_boundary_a"));
    REQUIRE_FALSE(contains_name_prefix(boundary_names, "cost_ed_phase_boundary_b"));
    REQUIRE(contains_name_prefix(boundary_names, "cost_st_phase_boundary_b"));
    REQUIRE_FALSE(contains_name_prefix(tail_names, "cost_st_phase_boundary_b"));
    REQUIRE(contains_name_prefix(tail_names, "cost_ed_phase_boundary_b"));

    const auto &prev_end_cost = require_func_named_prefix(flat.at(1)->problem_ptr(), __cost, "cost_ed_phase_boundary_a");
    REQUIRE(prev_end_cost.in_args().front()->field() == __y);
    const auto &boundary_cost = require_func_named_prefix(flat.at(1)->problem_ptr(), __cost, "cost_st_phase_boundary_b");
    REQUIRE(boundary_cost.in_args().front()->field() == __y);
}

TEST_CASE("phase-boundary endpoint terms survive current-interval inactive arguments", "[graph][mapping]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_phase_enable_boundary", 1);
    auto u = sym::inputs("u_phase_enable_boundary", 1);

    auto stage_a = stage_ocp::create();
    stage_a->add(*layout_cost("cost_phase_enable_a_u", var_list{u}));

    auto stage_b = stage_ocp::create();
    stage_b->add(*layout_dynamics("dyn_phase_enable_b", var_list{x, xn, u}, expr_dim(xn)));
    auto enabled_endpoint = layout_constr("start_phase_enable_boundary", var_list{x}, __eq_x, expr_dim(x));
    enabled_endpoint->enable_if_all({u});
    stage_b->st().add(*enabled_endpoint);

    ns_sqp sqp;
    sqp.add_stage(stage_a->clone(ocp::active_status_config{{u}, {}}), 1);
    sqp.add_stage(stage_b, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 2);
    REQUIRE(flat.front()->problem().dim(__u) == 0);
    const auto &lowered = require_func_named_prefix(flat.front()->problem_ptr(), __eq_x, "start_phase_enable_boundary");
    REQUIRE(lowered.in_args().front()->field() == __y);
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __eq_x), "start_phase_enable_boundary"));
}

TEST_CASE("sqp add_stages can append from an explicit end-node view", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_explicit_append", 1);
    auto u = sym::inputs("u_explicit_append", 1);
    auto stage_a = make_stage("explicit_a", x, xn, u);
    auto stage_b = make_stage("explicit_b", x, xn, u);
    stage_b->st().add(*layout_cost("cost_st_explicit_b", var_list{x}));
    ns_sqp sqp;
    auto first = sqp.add_stage(stage_a, 1);
    sqp.add_stages(first.back()->ed(), stage_b, 2);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_x_explicit_a"));
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_st_explicit_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_x_explicit_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_st_explicit_b"));
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_st_explicit_b"));

    const auto &boundary_cost = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_st_explicit_b");
    REQUIRE(boundary_cost.in_args().front()->field() == __y);
    const auto &repeat_boundary_cost = require_func_named_prefix(flat.at(1)->problem_ptr(), __cost, "cost_st_explicit_b");
    REQUIRE(repeat_boundary_cost.in_args().front()->field() == __y);
}

TEST_CASE("sqp add_stages rejects disconnected first start nodes", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_disconnected_start", 1);
    auto u = sym::inputs("u_disconnected_start", 1);
    auto stage = make_stage("disconnected_start", x, xn, u);

    ns_sqp sqp;
    REQUIRE_THROWS_WITH(
        sqp.add_stages(stage->st(), stage, 1),
        Catch::Matchers::ContainsSubstring("first path must start from sqp.start_node"));
}

TEST_CASE("sqp add_stages supports multiple explicit successors from one boundary", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_multi_successor", 1);
    auto u = sym::inputs("u_multi_successor", 1);
    auto stage_a = make_stage("multi_successor_a", x, xn, u);
    auto stage_b = make_stage("multi_successor_b", x, xn, u);
    auto stage_c = make_stage("multi_successor_c", x, xn, u);
    stage_b->st().add(*layout_cost("cost_st_multi_successor_b", var_list{x}));
    stage_c->st().add(*layout_cost("cost_st_multi_successor_c", var_list{x}));

    ns_sqp sqp;
    auto first = sqp.add_stage(stage_a, 1);
    sqp.add_stages(first.back()->ed(), stage_b, 1);
    sqp.add_stages(first.back()->ed(), stage_c, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    const auto first_names = expr_names(flat.front()->problem(), __cost);
    REQUIRE(contains_name_prefix(first_names, "cost_st_multi_successor_b"));
    REQUIRE(contains_name_prefix(first_names, "cost_st_multi_successor_c"));
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_st_multi_successor_b"));
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_st_multi_successor_c"));

    const auto &b_start = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_st_multi_successor_b");
    const auto &c_start = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_st_multi_successor_c");
    REQUIRE(b_start.in_args().front()->field() == __y);
    REQUIRE(c_start.in_args().front()->field() == __y);
}

TEST_CASE("sqp add_stages from explicit graph start preserves start-node terms", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_explicit_graph_start", 1);
    auto u = sym::inputs("u_explicit_graph_start", 1);
    auto stage_a = make_stage("explicit_graph_start_a", x, xn, u);
    auto stage_b = make_stage("explicit_graph_start_b", x, xn, u);

    ns_sqp sqp;
    sqp.start_node().add(*layout_cost("cost_explicit_graph_start", var_list{x}));
    sqp.add_stage(stage_a, 1);
    sqp.add_stages(sqp.start_node(), stage_b, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 2);
    const auto &first_start_cost = require_func_named_prefix(
        flat.front()->problem_ptr(), __cost, "cost_explicit_graph_start");
    const auto &branch_start_cost = require_func_named_prefix(
        flat.back()->problem_ptr(), __cost, "cost_explicit_graph_start");
    REQUIRE(first_start_cost.in_args().front()->field() == __x);
    REQUIRE(branch_start_cost.in_args().front()->field() == __x);
}

TEST_CASE("returned graph-owned stage handles are mutable and invalidate the runtime cache", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_mutable_stage", 1);
    auto u = sym::inputs("u_mutable_stage", 1);
    auto stage = make_stage("mutable_stage", x, xn, u);

    ns_sqp sqp;
    auto stages = sqp.add_stage(stage, 1);
    REQUIRE_FALSE(contains_name_prefix(expr_names(sqp.solver_nodes().front()->problem(), __cost), "cost_added_to_owned_stage"));

    stages.front()->ed().add(*layout_cost("cost_added_to_owned_stage", var_list{x}));

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 1);
    const auto &terminal_cost = require_func_named_prefix(flat.front()->problem_ptr(), __cost, "cost_added_to_owned_stage");
    REQUIRE(terminal_cost.in_args().front()->field() == __y);
}

TEST_CASE("adding an existing expression to a new endpoint role invalidates the runtime cache", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_mutable_role", 1);
    auto u = sym::inputs("u_mutable_role", 1);
    auto stage = make_stage("mutable_role", x, xn, u);
    auto boundary_cost = layout_cost("cost_mutable_role_boundary", var_list{x});

    ns_sqp sqp;
    auto stages = sqp.add_stage(stage, 1);
    stages.front()->st().add(*boundary_cost);

    REQUIRE_FALSE(contains_name_prefix(
        expr_names(sqp.solver_nodes().front()->problem(), __cost),
        "cost_mutable_role_boundary"));

    stages.front()->ed().add(*boundary_cost);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 1);
    const auto &lowered = require_func_named_prefix(
        flat.front()->problem_ptr(), __cost, "cost_mutable_role_boundary");
    REQUIRE(lowered.in_args().front()->field() == __y);
}

TEST_CASE("stage prototype mutation after add_stage does not affect graph-owned clones", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_formulation_dirty", 1);
    auto u = sym::inputs("u_formulation_dirty", 1);
    auto stage = make_stage("formulation_dirty", x, xn, u);

    ns_sqp sqp;
    sqp.add_stage(stage, 1);

    auto &flat_first = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_first.front()->problem(), __cost), "cost_added_after_realize"));

    stage->add(*layout_cost("cost_added_after_realize", var_list{x}));

    auto &flat_after_mutation = sqp.solver_nodes();
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat_after_mutation.front()->problem(), __cost), "cost_added_after_realize"));
}

TEST_CASE("stage clone active status is honored during composition", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_stage_active", 1);
    auto ua = sym::inputs("u_stage_keep", 1);
    auto ub = sym::inputs("u_stage_drop", 1);

    auto stage = stage_ocp::create();
    stage->add(*layout_dynamics("dyn_stage_active", var_list{x, xn, ua, ub}, expr_dim(xn)));
    stage->add(*layout_cost("cost_stage_active_u", var_list{ua, ub}));

    ns_sqp sqp;
    auto active_stage = stage->clone(ocp::active_status_config{{ub}, {}});
    sqp.add_stage(active_stage, 1);

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 1);
    REQUIRE(flat.front()->problem().dim(__u) == 1);
    REQUIRE(expr_names(flat.front()->problem(), __u) == std::vector<std::string>{"u_stage_keep"});
}

TEST_CASE("stage and node_view reject invalid endpoint placement", "[graph][validation]") {
    using namespace moto;

    auto [x, y] = sym::states("node_guard_x", 1);
    auto u = sym::inputs("node_guard_u", 1);
    auto stage = stage_ocp::create();

    auto x_only = layout_cost("x_only_cost", var_list{x});
    REQUIRE_NOTHROW(stage->add(*x_only));
    REQUIRE_THROWS_WITH(
        stage->ed().add(*x_only),
        Catch::Matchers::ContainsSubstring("both interval and endpoint"));
    REQUIRE_THROWS_WITH(
        stage->add(*layout_cost("y_only_cost", var_list{y})),
        Catch::Matchers::ContainsSubstring("pure y-only terms"));
    REQUIRE_NOTHROW(stage->add(*layout_dynamics("stage_guard_dyn", var_list{x, y, u}, expr_dim(y))));
    REQUIRE_NOTHROW(stage->ed().add(*layout_cost("ed_x_only_cost", var_list{x})));
    REQUIRE_THROWS_WITH(
        stage->ed().add(*layout_cost("ed_u_cost", var_list{u})),
        Catch::Matchers::ContainsSubstring("node_view only accepts terms"));
    REQUIRE_THROWS_WITH(
        stage->ed().add(*layout_cost("ed_y_cost", var_list{y})),
        Catch::Matchers::ContainsSubstring("node_view only accepts terms"));

    auto endpoint_stage = stage_ocp::create();
    auto endpoint_only = layout_cost("endpoint_only_cost", var_list{x});
    REQUIRE_NOTHROW(endpoint_stage->st().add(*endpoint_only));
    REQUIRE_NOTHROW(endpoint_stage->ed().add(*endpoint_only));
    REQUIRE_THROWS_WITH(
        endpoint_stage->add(*endpoint_only),
        Catch::Matchers::ContainsSubstring("both interval and endpoint"));
}

TEST_CASE("ocp active status can reactivate disabled expressions", "[graph][validation]") {
    using namespace moto;

    auto x = sym::states("x_active_reactivate", 1).first;
    auto x_cost = layout_cost("cost_active_reactivate", var_list{x});
    auto stage = stage_ocp::create();
    stage->add(*x_cost);

    stage->update_active_status({{*x_cost}, {}});
    REQUIRE_FALSE(stage->is_active(*x_cost));
    REQUIRE_NOTHROW(stage->update_active_status({{}, {*x_cost}}));
    REQUIRE(stage->is_active(*x_cost));
}

TEST_CASE("optimized initial state uses an internal virtual stage without exposing it", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_initial_state_opt", 1);
    auto u = sym::inputs("u_initial_state_opt", 1);
    constexpr scalar_t target = 2.0;

    constexpr size_t n_stages = 3;

    auto configure_solver = [&](ns_sqp &sqp) {
        auto stage = stage_ocp::create();
        stage->add(*callback_linear_dynamics("dyn_initial_state_opt", x, xn, u));
        stage->add(*callback_quadratic_cost("cost_initial_state_input", u));
        sqp.add_stage(stage, n_stages);
        sqp.start_node().add(*callback_quadratic_cost("cost_initial_state_target", x, target));
        sqp.settings.restoration.enabled = false;
        sqp.settings.prim_tol = 1e-8;
        sqp.settings.dual_tol = 1e-8;
        sqp.settings.comp_tol = 1e-8;
    };

    ns_sqp fixed;
    configure_solver(fixed);
    {
        auto &nodes = fixed.solver_nodes();
        REQUIRE(nodes.size() == n_stages);
        nodes.front()->sym_val().value_[__x].setZero();
        nodes.front()->sym_val().value_[__y].setZero();
        nodes.front()->sym_val().value_[__u].setZero();
    }
    REQUIRE(std::abs(fixed.solver_nodes().front()->sym_val().value_[__x](0)) < 1e-12);

    ns_sqp optimized;
    configure_solver(optimized);
    optimized.settings.initial_state = ns_sqp::initial_state_mode::optimized;
    {
        auto &nodes = optimized.solver_nodes();
        REQUIRE(nodes.size() == n_stages);
        nodes.front()->sym_val().value_[__x].setZero();
        nodes.front()->sym_val().value_[__y].setZero();
        nodes.front()->sym_val().value_[__u].setZero();
    }
    const auto result = optimized.update(10, false);
    REQUIRE(result.iter.result == ns_sqp::iter_result_t::success);
    REQUIRE(std::abs(optimized.solver_nodes().front()->sym_val().value_[__x](0) - target) < 1e-6);
}
