#include <arm/costs.hpp>
#include <arm/dynamics.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <atri/utils/print.hpp>

int main() {
    using namespace atri;
    auto prob = problem::make();
    armDynamics dyn;
    prob->add(dyn.dyn_pos);
    prob->add(dyn.dyn_vel);
    prob->add(armCosts::running(dyn.q, dyn.v, dyn.tau));
    auto prob_terminal = prob->copy();
    prob_terminal->add(armCosts::terminal(dyn.q_next, dyn.v_next));
    utils::print_problem(prob);
    fmt::print("good job!\n");

    ns_sqp sqp;

    auto init_node = sqp.graph_.add(ns_sqp::node_type(prob));
    auto end_node = sqp.graph_.add(ns_sqp::node_type(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 5);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);
    init_node->value(dyn.q).setOnes();

    sqp.graph_.apply_all_unary_parallel([](ns_sqp::node_type *node) {
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

    sqp.graph_.apply_all_unary_forward([&dyn](ns_sqp::node_type *node) {
        auto &data = ns_riccati::get_data(node);
        // std::cout << "delX  " << data.rollout_->prim_[__x].transpose() << '\n';
        // std::cout << magic_enum::enum_name(data.rank_status_) << '\n';
        std::cout << "state " << data.sym_->value_[__x].transpose() << '\n';
        std::cout << "input " << data.sym_->value_[__u].transpose() << '\n';
        // std::cout << "nexts " << data.sym_->value_[__y].transpose() << '\n';
        // std::cout << "rescs " << node->data(dyn.vel_zero_constr).v_.transpose() << '\n';
        // std::cout << "dual  " << static_cast<constr_data &>(node->data(dyn.vel_zero_constr)).multiplier_.transpose() << '\n';
        // std::cout << "sy    " << data.nsp_->s_y << '\n';
        // std::cout << "su    " << data.nsp_->s_u << '\n';
        // std::cout << "resdy " << data.dense_->approx_[__dyn].v_.transpose() << '\n';
        // std::cout << "jacdy \n" << data.dense_->approx_[__dyn].jac_[__y] << '\n';
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

    fmt::print("well done!\n");
    return 0;
}