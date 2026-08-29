#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/ocp/graph_model.hpp>
#include <moto/ocp/graph_composer.hpp>
#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ns_sqp.hpp>

namespace {
const bool force_sync_codegen_for_test = []() {
    setenv("MOTO_SYNC_CODEGEN", "1", 1);
    return true;
}();

std::vector<std::string> expr_names(const moto::ocp_base &prob, moto::field_t field) {
    std::vector<std::string> names;
    for (const moto::expr_handle &expr : prob.exprs(field)) {
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
    return v->dim();
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
    auto it = std::find_if(prob->exprs(field).begin(), prob->exprs(field).end(), [&](const moto::expr_handle &expr) {
        return expr->name().rfind(prefix, 0) == 0;
    });
    REQUIRE(it != prob->exprs(field).end());
    const auto *func = dynamic_cast<const moto::generic_func *>((*it).get());
    REQUIRE(func != nullptr);
    return *func;
}

template <typename Work>
std::pair<double, double> timing_percentiles(size_t trials, Work work) {
    using clock = std::chrono::steady_clock;
    std::vector<double> elapsed;
    elapsed.reserve(trials);
    bool valid = true;
    for (size_t trial = 0; trial < trials; ++trial) {
        const auto start = clock::now();
        valid &= work(trial);
        elapsed.push_back(std::chrono::duration<double, std::micro>(
                              clock::now() - start).count());
    }
    REQUIRE(valid);
    std::sort(elapsed.begin(), elapsed.end());
    return {elapsed[trials / 2], elapsed[trials * 95 / 100]};
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
    for (size_t i = 0; i < 3; ++i)
        sqp.stages().push_back(stage->copy());

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
    for (size_t i = 0; i < 2; ++i)
        sqp.stages().push_back(stage->copy());

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
    for (size_t i = 0; i < 4; ++i)
        sqp.stages().push_back(stage->copy());

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
    for (size_t i = 0; i < 4; ++i)
        sqp.stages().push_back(stage->copy());

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

    auto first = c->reuse_remap({{x_a, y_a}});
    auto second = c->reuse_remap({{x_a, y_a}});
    auto redundant = c->reuse_remap({{x_a, y_a}, {x_a, y_a}});
    auto identity = c->reuse_remap({{x_a, x_a}});
    auto different = c->reuse_remap({{x_a, y_b}});

    REQUIRE(first.get() == second.get());
    REQUIRE(first.get() == redundant.get());
    REQUIRE(first->finalized());
    REQUIRE(first->wait_until_ready());
    REQUIRE(identity.get() == c.get());
    REQUIRE(identity->finalized());
    REQUIRE(first.get() != different.get());
    REQUIRE(expr_cast<generic_func>(first)->in_args().front()->uid() == y_a->uid());
    REQUIRE(expr_cast<generic_func>(first)->arg_num(__x) == 0);
    REQUIRE(expr_cast<generic_func>(first)->arg_num(__y) == 1);
    REQUIRE(expr_cast<generic_func>(different)->in_args().front()->uid() == y_b->uid());
    const auto fresh_a = c->remap_arguments({{x_a, y_a}});
    const auto fresh_b = c->remap_arguments({{x_a, y_a}});
    REQUIRE(fresh_a.get() != fresh_b.get());
    REQUIRE(fresh_a->uid() != fresh_b->uid());
    REQUIRE(fresh_a->finalized());
    REQUIRE(fresh_a->wait_until_ready());
    REQUIRE(fresh_a->handle().get() == fresh_a.get());
    REQUIRE_THROWS_AS(c->reuse_remap({{x_a, p}}), std::runtime_error);
    REQUIRE_THROWS_AS(c->reuse_remap({{p, u_remap}}), std::runtime_error);
    auto y_bad_dim = sym::states("x_remap_bad_dim", 2).second;
    auto p_bad_dim = sym::params("p_remap_bad_dim", 2);
    REQUIRE_THROWS_AS(c->reuse_remap({{x_a, y_bad_dim}}), std::runtime_error);
    REQUIRE_THROWS_AS(c->reuse_remap({{p, p_bad_dim}}), std::runtime_error);

    std::vector<expr_handle> threaded_results(16);
    std::vector<std::thread> remap_threads;
    for (size_t i = 0; i < threaded_results.size(); ++i) {
        remap_threads.emplace_back([&, i]() {
            threaded_results[i] = c->reuse_remap({{x_a, y_a}});
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

    auto remapped_dyn = dyn->remap_arguments({});
    REQUIRE(dynamic_cast<dense_dynamics *>(remapped_dyn.get()) != nullptr);

    REQUIRE(ineq->finalize(true));
    REQUIRE(ineq->get_codegen_task() != nullptr);
    auto remapped = ineq->remap_arguments({{x, y}});
    REQUIRE(dynamic_cast<ineq_constr *>(remapped.get()) != nullptr);
    REQUIRE(expr_cast<generic_func>(remapped)->get_codegen_task() == nullptr);
    REQUIRE(expr_cast<generic_func>(remapped)->in_args().front()->uid() == y->uid());
}

TEST_CASE("expression and endpoint handles have explicit identity semantics",
          "[graph][handle]") {
    using namespace moto;

    auto x = sym::state("x_handle_identity", 1);
    const auto same = x->handle();
    const auto independent = x->clone("x_handle_independent");
    REQUIRE(same.get() == x.get());
    REQUIRE(same->uid() == x->uid());
    REQUIRE(independent->uid() != x->uid());
    REQUIRE(independent->name() == "x_handle_independent");

    auto prototype = stage_ocp::create();
    auto shared_cost = layout_cost("stage_copy_shared_cost", var_list{x});
    prototype->add(*shared_cost);
    auto copied = prototype->copy();
    REQUIRE(copied->exprs(__cost).front().get() == shared_cost.get());
    copied->add(*layout_cost("stage_copy_local_cost", var_list{x}));
    REQUIRE(prototype->num(__cost) == 1);
    REQUIRE(copied->num(__cost) == 2);

    node_view endpoint;
    {
        auto stage = stage_ocp::create();
        endpoint = stage->ed();
    }
    REQUIRE(bool(endpoint));
    REQUIRE(endpoint.stage() != nullptr);
    REQUIRE_NOTHROW(endpoint.add(*layout_cost(
        "endpoint_owned_handle_cost", var_list{x})));
}

TEST_CASE("sqp stages is the graph-owned stage vector", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_append_stage", 1);
    auto u = sym::inputs("u_append_stage", 1);
    auto stage_a = make_stage("append_a", x, xn, u);
    auto stage_b = make_stage("append_b", x, xn, u);

    ns_sqp sqp;
    sqp.stages().push_back(stage_a->copy());

    REQUIRE(sqp.solver_nodes().size() == 1);

    sqp.stages().push_back(stage_b->copy());
    sqp.stages().push_back(stage_b->copy());
    REQUIRE(sqp.stages().size() == 3);
    REQUIRE(sqp.st().stage() == sqp.start_node().stage());
    REQUIRE(sqp.ed().stage() != sqp.stages().back());

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 3);
    REQUIRE(contains_name_prefix(expr_names(flat.front()->problem(), __cost), "cost_u_append_a"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_u_append_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.at(1)->problem(), __cost), "cost_x_append_b"));
    REQUIRE(contains_name_prefix(expr_names(flat.back()->problem(), __cost), "cost_x_append_b"));
    const auto &stage_cost = require_func_named_prefix(flat.back()->problem_ptr(), __cost, "cost_x_append_b");
    REQUIRE(stage_cost.in_args().front()->field() == __x);
}

TEST_CASE("appending stage copies preserves unchanged runtime nodes", "[graph][path][cache]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_append_copies", 1);
    auto u = sym::inputs("u_append_copies", 1);
    auto stage_a = make_stage("phases_a", x, xn, u);
    auto stage_b = make_stage("phases_b", x, xn, u);

    ns_sqp sqp;
    sqp.stages().push_back(stage_a->copy());
    sqp.stages().push_back(stage_a->copy());
    sqp.stages().push_back(stage_b->copy());

    auto &initial = sqp.solver_nodes();
    REQUIRE(initial.size() == 3);
    auto *unchanged = initial.front();
    unchanged->sym_val().value_[__x].setConstant(3.0);

    sqp.ed().add(*layout_cost("cost_phases_terminal", var_list{x}));
    auto &updated = sqp.solver_nodes();
    REQUIRE(updated.front() == unchanged);
    REQUIRE(updated.front()->sym_val().value_[__x](0) == 3.0);
    REQUIRE(contains_name_prefix(expr_names(updated.back()->problem(), __cost),
                                 "cost_phases_terminal"));
}

TEST_CASE("stage vector order lowers the next phase start endpoint onto the previous tail", "[graph][mapping]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_phase_boundary", 1);
    auto u = sym::inputs("u_phase_boundary", 1);
    auto stage_a = make_stage("phase_boundary_a", x, xn, u);
    auto stage_b = make_stage("phase_boundary_b", x, xn, u);
    stage_a->ed().add(*layout_cost("cost_ed_phase_boundary_a", var_list{x}));
    stage_b->st().add(*layout_cost("cost_st_phase_boundary_b", var_list{x}));
    stage_b->ed().add(*layout_cost("cost_ed_phase_boundary_b", var_list{x}));

    ns_sqp sqp;
    sqp.stages().push_back(stage_a->copy());
    sqp.stages().push_back(stage_a->copy());
    sqp.stages().push_back(stage_b->copy());

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
    sqp.stages().push_back(stage_a->copy(ocp::active_status_config{{u}, {}}));
    sqp.stages().push_back(stage_b->copy());

    auto &flat = sqp.solver_nodes();
    REQUIRE(flat.size() == 2);
    REQUIRE(flat.front()->problem().dim(__u) == 0);
    const auto &lowered = require_func_named_prefix(flat.front()->problem_ptr(), __eq_x, "start_phase_enable_boundary");
    REQUIRE(lowered.in_args().front()->field() == __y);
    REQUIRE_FALSE(contains_name_prefix(expr_names(flat.back()->problem(), __eq_x), "start_phase_enable_boundary"));
}

TEST_CASE("endpoint lowering maps the source primal mask from x to y", "[graph][mapping]") {
    using namespace moto;

    auto [xa, ya] = sym::states("x_lowered_status_a", 1);
    auto [xb, yb] = sym::states("x_lowered_status_b", 1);
    auto stage = stage_ocp::create();
    stage->ed().add(*layout_cost("cost_lowered_status", var_list{xa, xb}));

    ns_sqp sqp;
    sqp.stages().push_back(stage->copy(ocp::active_status_config{{xb}, {}}));

    const auto &problem = sqp.solver_nodes().front()->problem();
    REQUIRE(problem.dim(__y) == 1);
    REQUIRE(problem.is_active(*ya));
    REQUIRE(problem.contains(*yb));
    REQUIRE_FALSE(problem.is_active(*yb));
    REQUIRE(contains_name_prefix(expr_names(problem, __cost), "cost_lowered_status"));
}

TEST_CASE("composed terms keep source-resolved predicates across copies", "[graph][mapping]") {
    using namespace moto;

    auto [x, y] = sym::states("x_resolved_endpoint_status", 1);
    auto gate = sym::inputs("u_resolved_endpoint_gate", 1);
    auto live = sym::inputs("u_resolved_endpoint_live", 1);

    auto current = stage_ocp::create();
    current->add(*layout_cost("cost_resolved_endpoint_gate", var_list{gate}));
    current->add(*layout_cost("cost_resolved_endpoint_live", var_list{live}));

    auto next = stage_ocp::create();
    next->add(*layout_dynamics("dyn_resolved_endpoint_status", var_list{x, y, gate, live}, expr_dim(y)));
    auto endpoint = layout_constr("constr_resolved_endpoint_status", var_list{x}, __eq_x, expr_dim(x));
    endpoint->enable_if_all({gate});
    next->st().add(*endpoint);

    ns_sqp sqp;
    sqp.stages().push_back(current->copy(ocp::active_status_config{{gate}, {}}));
    sqp.stages().push_back(next->copy());

    const auto source = sqp.solver_nodes().front()->problem_ptr();
    REQUIRE(source->dim(__u) == 1);
    REQUIRE(contains_name_prefix(expr_names(*source, __eq_x), "constr_resolved_endpoint_status"));

    const auto &live_cost = require_func_named_prefix(source, __cost, "cost_resolved_endpoint_live");
    ocp::active_status_config deactivate_unrelated;
    deactivate_unrelated.deactivate_list.emplace_back(live_cost.handle());
    auto copied = source->copy(deactivate_unrelated);
    copied->wait_until_ready();
    REQUIRE(contains_name_prefix(expr_names(*copied, __eq_x), "constr_resolved_endpoint_status"));

    ocp::active_status_config deactivate_argument;
    deactivate_argument.deactivate_list.emplace_back(y->handle());
    auto without_argument = source->copy(deactivate_argument);
    without_argument->wait_until_ready();
    REQUIRE_FALSE(contains_name_prefix(expr_names(*without_argument, __eq_x), "constr_resolved_endpoint_status"));
}

TEST_CASE("authored endpoint handles access lowered runtime data", "[graph][mapping]") {
    using namespace moto;

    auto [x, y] = sym::states("x_endpoint_access", 1);
    auto endpoint = layout_constr("constr_endpoint_access", var_list{x}, __eq_x, 1);
    endpoint->value = [](func_approx_data &data) { data.v_ = data[0]; };
    endpoint->jacobian = [](func_approx_data &data) {
        data.jac_[0](0, 0) = 1.;
    };
    endpoint->hessian = [](func_approx_data &) {};

    auto stage = stage_ocp::create();
    stage->ed().add(*endpoint);
    ns_sqp sqp;
    sqp.stages().push_back(stage->copy());

    auto *node = sqp.solver_nodes().front();
    node->sym_val()[y](0) = 3.25;
    node->update_approximation(node_data::update_mode::eval_all);

    auto &runtime = static_cast<node_data &>(*node).data(endpoint);
    REQUIRE(std::abs(runtime.v_(0) - 3.25) < 1e-12);
    const sym &source_x = x;
    REQUIRE(runtime[source_x].data() == node->sym_val()[y].data());
    REQUIRE(std::abs(runtime.jac(source_x)(0, 0) - 1.) < 1e-12);
}

TEST_CASE("appending stage copies advances the current end boundary", "[graph][path]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_explicit_append", 1);
    auto u = sym::inputs("u_explicit_append", 1);
    auto stage_a = make_stage("explicit_a", x, xn, u);
    auto stage_b = make_stage("explicit_b", x, xn, u);
    stage_b->st().add(*layout_cost("cost_st_explicit_b", var_list{x}));
    ns_sqp sqp;
    sqp.stages().push_back(stage_a->copy());
    sqp.stages().push_back(stage_b->copy());
    sqp.stages().push_back(stage_b->copy());

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

TEST_CASE("native stage vector supports replacement and horizon shift", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_replace_stages", 1);
    auto u = sym::inputs("u_replace_stages", 1);
    auto stage_a = make_stage("replace_a", x, xn, u);
    auto stage_b = make_stage("replace_b", x, xn, u);
    auto stage_c = make_stage("replace_c", x, xn, u);

    ns_sqp sqp;
    for (size_t i = 0; i < 5; ++i)
        sqp.stages().push_back(stage_a->copy());
    auto &stages = sqp.stages();
    const auto keep_0 = stages[0];
    const auto keep_4 = stages[4];
    auto graph_end = sqp.ed();
    graph_end.add(*layout_cost("cost_replace_stages_terminal", var_list{x}));

    stages.erase(stages.begin() + 1, stages.begin() + 3);
    stages.insert(stages.begin() + 1, stage_b->copy());
    REQUIRE(sqp.stages()[0] == keep_0);
    REQUIRE(sqp.stages()[3] == keep_4);
    REQUIRE(contains_name_prefix(expr_names(sqp.solver_nodes()[1]->problem(), __cost), "cost_x_replace_b"));

    auto before_shift = sqp.solver_nodes();
    before_shift[2]->sym_val().value_[__x].setConstant(7.0);
    const auto shifted_0 = stages[1];
    const auto shifted_1 = stages[2];
    stages.erase(stages.begin());
    stages.push_back(stage_c->copy());
    REQUIRE(sqp.stages()[0] == shifted_0);
    REQUIRE(sqp.stages()[1] == shifted_1);
    auto &after_shift = sqp.solver_nodes();
    REQUIRE(after_shift[1] == before_shift[2]);
    REQUIRE(after_shift[1]->sym_val().value_[__x](0) == 7.0);
    REQUIRE(contains_name_prefix(expr_names(after_shift.back()->problem(), __cost), "cost_x_replace_c"));
    REQUIRE(contains_name_prefix(expr_names(after_shift.back()->problem(), __cost), "cost_replace_stages_terminal"));
    REQUIRE(sqp.ed().stage() == graph_end.stage());
}

TEST_CASE("set_stages rejects aliased stage entries", "[graph][validation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_aliased_stages", 1);
    auto u = sym::inputs("u_aliased_stages", 1);
    auto stage = make_stage("aliased_stages", x, xn, u);
    ns_sqp sqp;
    sqp.stages().push_back(stage->copy());
    auto &stages = sqp.stages();
    stages.push_back(stages[0]);
    REQUIRE_THROWS_WITH(
        sqp.solver_nodes(),
        Catch::Matchers::ContainsSubstring("distinct stage objects"));
}

TEST_CASE("500-stage graph construction and shift stay within the pure graph budget",
          "[graph][performance]") {
    using namespace moto;

    auto prototype = stage_ocp::create();
    std::vector<stage_ocp_ptr_t> prepared;
    prepared.reserve(500);
    for (size_t i = 0; i < 500; ++i)
        prepared.push_back(prototype->copy());

    const auto [construction_median, construction_p95] = timing_percentiles(100, [&](size_t) {
        graph_model candidate;
        graph_composer candidate_composer;
        candidate.stages().insert(candidate.stages().end(), prepared.begin(), prepared.end());
        return candidate_composer.compose(candidate).intervals->size() == 500;
    });
    REQUIRE(construction_median < 50.0);
    REQUIRE(construction_p95 < 50.0);

    graph_model graph;
    graph_composer composer;
    for (size_t i = 0; i < 500; ++i)
        graph.stages().push_back(prototype->copy());

    const auto initial = composer.compose(graph);
    REQUIRE(initial.intervals->size() == 500);
    REQUIRE(initial.intervals->front().formulation != initial.intervals->at(1).formulation);
    REQUIRE(initial.intervals->at(1).formulation == initial.intervals->at(2).formulation);
    REQUIRE(initial.intervals->back().formulation != initial.intervals->at(1).formulation);

    std::vector<stage_ocp_ptr_t> tails;
    for (size_t i = 0; i < 200; ++i)
        tails.push_back(prototype->copy());
    const auto [median, p95] = timing_percentiles(200, [&](size_t trial) {
        graph.stages().erase(graph.stages().begin());
        graph.stages().push_back(std::move(tails[trial]));
        return composer.compose(graph).intervals->size() == 500;
    });
#ifdef NDEBUG
    REQUIRE(median < 50.0);
    REQUIRE(p95 < 50.0);
#endif
}

TEST_CASE("returned graph-owned stage handles are mutable and invalidate the runtime cache", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_mutable_stage", 1);
    auto u = sym::inputs("u_mutable_stage", 1);
    auto stage = make_stage("mutable_stage", x, xn, u);

    ns_sqp sqp;
    sqp.stages().push_back(stage->copy());
    auto &stages = sqp.stages();
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
    sqp.stages().push_back(stage->copy());
    auto &stages = sqp.stages();
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

TEST_CASE("stage prototype mutation after insertion does not affect graph-owned clones", "[graph][mutation]") {
    using namespace moto;

    auto [x, xn] = sym::states("x_formulation_dirty", 1);
    auto u = sym::inputs("u_formulation_dirty", 1);
    auto stage = make_stage("formulation_dirty", x, xn, u);

    ns_sqp sqp;
    sqp.stages().push_back(stage->copy());

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
    auto active_stage = stage->copy(ocp::active_status_config{{ub}, {}});
    sqp.stages().push_back(active_stage->copy());

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
        Catch::Matchers::ContainsSubstring("endpoint only accepts terms"));
    REQUIRE_THROWS_WITH(
        stage->ed().add(*layout_cost("ed_y_cost", var_list{y})),
        Catch::Matchers::ContainsSubstring("endpoint only accepts terms"));

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
        for (size_t i = 0; i < n_stages; ++i)
            sqp.stages().push_back(stage->copy());
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
