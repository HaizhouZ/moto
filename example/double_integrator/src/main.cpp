#include <atri/core/directed_graph.hpp>
#include <atri/ocp/core/approx_data.hpp>
#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/data/rollout_data.hpp>
#include <atri/solver/fwd.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>
#include <iostream>

using namespace atri;

int main() {
    doubleIntegratorDyn dyn;
    doubleIntegratorCosts costs;
    auto prob = std::make_shared<problem>();
    prob->add(dyn);
    prob->add(costs.running(dyn.r, dyn.v, dyn.a));
    auto prob_terminal = prob->copy();
    prob_terminal->add(costs.terminal(dyn.r_next, dyn.v_next));

    utils::print_problem(prob);

    fmt::print("good!\n");

    ns_sqp sqp;

    auto init_node = sqp.graph_.add(ns_sqp::node(prob));
    auto end_node = sqp.graph_.add(ns_sqp::node(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 10);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    init_node->get(dyn.r).setConstant(6);

    // auto start_time = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < 10000; i++) {
    //     sqp.update();
    // }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // fmt::print("Update took {} us\n", duration / 10000);

    // for (size_t i = 0; i < sqp.timings.size(); i++) {
    //     fmt::print("Timing[{}]: {}us\n", i, sqp.timings[i] / 10000);
    // }
    sqp.update(1);
    // auto &data = ns_riccati_solver::get_data(init_node.get());
    // std::cout << data.rollout_->prim_[__x].transpose() << '\n';
    // sqp.update();
    sqp.graph_.apply_all_unary_forward([](ns_sqp::node *node) {
        auto &data = ns_riccati_solver::get_data(node);
        // why the delta y is wrong?
        // std::cout << data.rollout_->prim_[__x].transpose() << '\n';
        std::cout << "state " << data.sym_->value_[__x].transpose() << '\n';
        std::cout << "input " << data.sym_->value_[__u].transpose() << '\n';
        std::cout << "nexts " << data.sym_->value_[__y].transpose() << '\n';
        std::cout << "resdy " << data.raw_->approx_[__dyn].v_.transpose() << '\n';
        std::cout << '\n';
        // std::cout << data.raw_->dual_[__dyn].transpose() << '\n';
        // std::cout << data.d_u.K.transpose() << '\n';
        // std::cout << data.nsp_->u_0_p_k.transpose() << '\n';
        // std::cout << data.nsp_->U << '\n';
        // std::cout << data.Q_y.transpose() << '\n' << '\n';
        // std::cout << data.Q_yy << '\n' << '\n';
        // std::cout << data.raw_->approx_[__dyn].v_[__x] << '\n' << '\n';
    });

    fmt::print("nice!\n");

    return 0;
}