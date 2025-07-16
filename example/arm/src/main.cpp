#include <arm/costs.hpp>
#include <arm/dynamics.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/print.hpp>
#include <iostream>

int main() {
    using namespace moto;
    auto prob = ocp::make();
    armDynamics dyn;
    prob->add(dyn.dyn_pos);
    prob->add(dyn.dyn_vel);
    prob->add(armCosts::running(dyn.q, dyn.v, dyn.tau));
    auto prob_terminal = prob->copy();
    prob_terminal->add(armCosts::terminal(dyn.q_next, dyn.v_next));
    utils::print_problem(prob);
    utils::print_problem(prob_terminal);
    fmt::print("good job!\n");

    ns_sqp sqp;

    auto& init_node = sqp.graph_.add(ns_sqp::node_type(prob));
    auto& end_node = sqp.graph_.add(ns_sqp::node_type(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 8);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);
    init_node->value(dyn.q).setOnes();

    sqp.graph_.apply_all_unary_parallel([](auto *node) {
        node->value(armCosts::ee_cost::r_des).setConstant(0.5);
        node->value(armCosts::ee_cost::W_kin).setConstant(100);
    });

    // for(auto i: range(1000)){
    //     sqp.forward();
    // }
    try {
        sqp.update(10);
    } catch (...) {
    }

    sqp.graph_.apply_all_unary_forward([&dyn](auto *node) {
        // std::cout << "delX  " << node->rollout_->prim_[__x].transpose() << '\n';
        // std::cout << magic_enum::enum_name(node->rank_status_) << '\n';
        std::cout << "state " << node->sym_->value_[__x].transpose() << '\n';
        std::cout << "input " << node->sym_->value_[__u].transpose() << '\n';
        std::cout << "nexts " << node->sym_->value_[__y].transpose() << '\n';
        std::cout << "param " << node->sym_->value_[__p].transpose() << '\n';
        // std::cout << "rescs " << node->data(dyn.vel_zero_constr).v_.transpose() << '\n';
        // std::cout << "dual  " << static_cast<constr_approx_map &>(node->data(dyn.vel_zero_constr)).multiplier_.transpose() << '\n';
        // std::cout << "sy    " << node->nsp_->s_y << '\n';
        // std::cout << "su    " << node->nsp_->s_u << '\n';
        // std::cout << "resdy " << node->dense_->approx_[__dyn].v_.transpose() << '\n';
        // std::cout << "jacdy \n" << node->dense_->approx_[__dyn].jac_[__y] << '\n';
        std::cout << "dual  " << node->dense_->dual_[__dyn].transpose() << '\n';
        //     std::cout << "Qx    " << node->Q_x << '\n';
        //     std::cout << "Qu    " << node->Q_u << '\n';
        //     std::cout << "Qy    " << node->Q_y << '\n';
        //     // std::cout << "gain \n" << node->d_y.k.transpose() << '\n';
        //     // std::cout << node->d_u.K.transpose() << '\n';
        //     // std::cout << node->nsp_->u_0_p_k.transpose() << '\n';
        //     // std::cout << node->nsp_->U << '\n';
        //     // std::cout << node->Q_y.transpose() << '\n' << '\n';
        //     // std::cout << node->Q_yx << '\n' << '\n';
        //     // std::cout << node->dense_->approx_[__dyn].v_[__x] << '\n' << '\n';
        // std::cout << '\n';
    });

    fmt::print("well done!\n");
    return 0;
}