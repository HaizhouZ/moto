#include <atri/core/directed_graph.hpp>
#include <atri/ocp/approx_storage.hpp>
#include <atri/ocp/shooting_node.hpp>
#include <atri/solver/data/nullspace_data.hpp>
#include <atri/solver/data/rollout_data.hpp>
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

    auto init_node = sqp.graph_.add(ns_sqp::node_type(prob));
    auto end_node = sqp.graph_.add(ns_sqp::node_type(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 100);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    init_node->value(dyn.r).setConstant(6);

    size_t n_iter = 1000;

    auto start_time = std::chrono::high_resolution_clock::now();
    sqp.update(n_iter);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    fmt::print("Update took {} us\n", duration / n_iter);
    // per step timing
    // for (size_t i = 0; i < sqp.timings.size(); i++) {
    //     fmt::print("Timing[{}]: {}us\n", i, sqp.timings[i] / n_iter);
    // }

    // auto &data = ns_riccati::get_data(init_node.get());
    // std::cout << data.rollout_->prim_[__x].transpose() << '\n';
    // sqp.update();
    sqp.graph_.apply_all_unary_forward([](ns_sqp::node_type *node) {
        auto &data = ns_riccati::get_data(node);
        //     // why the delta y is wrong?
        // std::cout << "delX  " << data.rollout_->prim_[__x].transpose() << '\n';
        // std::cout << "state " << data.sym_->value_[__x].transpose() << '\n';
        //     std::cout << "input " << data.sym_->value_[__u].transpose() << '\n';
        //     std::cout << "nexts " << data.sym_->value_[__y].transpose() << '\n';
        //     std::cout << "resdy " << data.raw_->approx_[__dyn].v_.transpose() << '\n';
        //     std::cout << "dual  " << data.raw_->dual_[__dyn].transpose() << '\n';
        //     std::cout << "Qx    " << data.Q_x << '\n';
        //     std::cout << "Qu    " << data.Q_u << '\n';
        //     std::cout << "Qy    " << data.Q_y << '\n';
        //     // std::cout << "gain \n" << data.d_y.k.transpose() << '\n';
        //     // std::cout << data.d_u.K.transpose() << '\n';
        //     // std::cout << data.nsp_->u_0_p_k.transpose() << '\n';
        //     // std::cout << data.nsp_->U << '\n';
        //     // std::cout << data.Q_y.transpose() << '\n' << '\n';
        //     // std::cout << data.Q_yx << '\n' << '\n';
        //     // std::cout << data.raw_->approx_[__dyn].v_[__x] << '\n' << '\n';
        //     std::cout << '\n';
    });

    fmt::print("nice!\n");

    return 0;
}