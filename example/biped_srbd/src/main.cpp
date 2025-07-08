#include <dynamics.hpp>
#include <iostream>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <moto/utils/print.hpp>

int main() {
    // moto::func_codegen_helper::enable();
    biped_srbd::srbd_dynamics dyn;
    std::cout << "Hello, Biped SRBD!" << std::endl;
    auto prob = moto::ocp::make();
    prob->add(dyn.euler());
    prob->add(dyn.friction_cone());
    prob->add(dyn.running_cost());
    prob->add(dyn.stance_foot_constr());
    prob->add(dyn.foot_loc_constr());
    auto terminal_prob = prob->copy();
    terminal_prob->add(dyn.terminal_cost());
    // moto::func_codegen_helper::wait_until_all_compiled(8);
    using namespace moto;
    utils::print_problem(prob);
    ns_sqp solver;
    auto &graph = solver.graph_;
    auto &init_node = graph.add(ns_sqp::node_type(prob));
    auto &end_node = graph.add(ns_sqp::node_type(terminal_prob));

    graph.add_edge(init_node, end_node, 100);

    init_node->value(dyn.r) << 0, 0, 0.5;     // initial position of the com
    init_node->value(dyn.r_l) << 0, 0.1, 0.;  // initial position of the left foot
    init_node->value(dyn.r_r) << 0, -0.1, 0.; // initial position of the right foot
    init_node->value(dyn.active_l).setOnes();
    init_node->value(dyn.active_r).setOnes();
    init_node->value(dyn.r_d) << 1.0, 0, 0.5; // desired position of the com

    return 0;
}