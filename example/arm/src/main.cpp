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

    sqp.graph_.add_edge(init_node, end_node, 200);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    // sqp.graph_.apply_all_unary_parallel([](ns_sqp::node_type *node) {
    //     node->value(armCosts::ee_cost::r_des).setConstant(0.3);
    // });

    // for(auto i: range(1000)){
    //     sqp.forward();
    // }
    sqp.update(2000);

    fmt::print("well done!\n");
    return 0;
}