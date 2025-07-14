#include <moto/core/directed_graph.hpp>
#include <moto/ocp/approx_storage.hpp>
#include <moto/ocp/shooting_node.hpp>
#include <moto/solver/nullspace_data.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/print.hpp>
#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>
#include <iostream>

using namespace moto;

int main() {
    doubleIntegratorDyn dyn;
    doubleIntegratorCosts costs;
    auto prob = ocp::make();
    prob->add(dyn);
    prob->add(costs.running(dyn.r, dyn.v, dyn.a));
    auto prob_terminal = prob->copy();
    prob_terminal->add(costs.terminal(dyn.r_next, dyn.v_next));

    utils::print_problem(prob);

    fmt::print("good!\n");

    ns_sqp sqp;

    auto& init_node = sqp.graph_.add(ns_sqp::node_type(prob));
    auto& end_node = sqp.graph_.add(ns_sqp::node_type(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 10);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    init_node->value(dyn.r).setConstant(6);
    init_node->value(dyn.v).setZero();

    size_t n_iter = 1;

    auto start_time = std::chrono::high_resolution_clock::now();
    sqp.update(n_iter);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    fmt::print("Update took {} us\n", duration / n_iter);
    // per step timing
    // for (size_t i = 0; i < sqp.timings.size(); i++) {
    //     fmt::print("Timing[{}]: {}us\n", i, sqp.timings[i] / n_iter);
    // }

    // auto &data = nullsp_kkt_solve::get_data(init_node.get());
    // std::cout << data.rollout_->prim_[__x].transpose() << '\n';
    // sqp.update();
    sqp.graph_.apply_all_unary_forward([&dyn](auto *node) {
        // std::cout << "delX  " << data.rollout_->prim_[__x].transpose() << '\n';
        // std::cout << magic_enum::enum_name(data.rank_status_) << '\n';
        // std::cout << "state " << node->sym_->value_[__x].transpose() << '\n';
        // std::cout << "input " << node->sym_->value_[__u].transpose() << '\n';
        std::cout << "inf_prim_res: " << node->inf_prim_res() << '\n';
        std::cout << "inf_dual_res: " << node->inf_dual_res() << '\n';
        // std::cout << "nexts " << data.sym_->value_[__y].transpose() << '\n';
        // std::cout << "rescs " << node->data(dyn.vel_zero_constr).v_.transpose() << '\n';
        // std::cout << "dual  " << static_cast<constr_data &>(node->data(dyn.vel_zero_constr)).multiplier_.transpose() << '\n';
        // std::cout << "sy    " << data.nsp_->s_y << '\n';
        // std::cout << "su    " << data.nsp_->s_u << '\n';
        //     std::cout << "resdy " << data.dense_->approx_[__dyn].v_.transpose() << '\n';
        //     std::cout << "dual  " << data.dense_->dual_[__dyn].transpose() << '\n';
        //     std::cout << "Qx    " << data.Q_x << '\n';
        //     std::cout << "Qu    " << data.Q_u << '\n';
        //     std::cout << "Qy    " << data.Q_y << '\n';
        //     // std::cout << "gain \n" << data.d_y.k.transpose() << '\n';
        //     // std::cout << data.d_u.K.transpose() << '\n';
        //     // std::cout << data.nsp_->u_0_p_k.transpose() << '\n';
        //     // std::cout << data.nsp_->U << '\n';
        //     // std::cout << data.Q_y.transpose() << '\n' << '\n';
        //     // std::cout << data.Q_yx << '\n' << '\n';
        //     // std::cout << data.dense_->approx_[__dyn].v_[__x] << '\n' << '\n';
        // std::cout << '\n';
    });

    fmt::print("nice!\n");

    return 0;
}