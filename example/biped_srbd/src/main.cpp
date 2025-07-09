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
    auto &init_node = graph.set_head(graph.add(ns_sqp::node_type(prob)));
    auto &end_node = graph.set_tail(graph.add(ns_sqp::node_type(terminal_prob)));

    graph.add_edge(init_node, end_node, 100); // 100 steps

    init_node->value(dyn.r) << 0, 0, 0.5;     // initial position of the com
    init_node->value(dyn.r_l) << 0, 0.1, 0.;  // initial position of the left foot
    init_node->value(dyn.r_r) << 0, -0.1, 0.; // initial position of the right foot
    init_node->value(dyn.active_l_cur).setOnes();
    init_node->value(dyn.active_r_cur).setOnes();
    init_node->value(dyn.r_d) << 1.0, 0, 0.5; // desired position of the com

    graph.apply_all_unary_parallel([&](node_data *data) {
        *data->sym_ = *init_node->sym_; // initialize symbolic data
    });
    // set gait
    int n = -10;
    int phase = -1; // -1 stance, 0 left swing, 1 right swing
    int steps = 0;
    graph.apply_all_unary_forward([&](node_data *data) {
        if (n >= 0) {
            if (steps < 3) {
                if (n % 25 == 0) { // every 25 steps, switch phase
                    if (phase == -1)
                        phase = 0;
                    else {
                        steps++;
                        if (steps < 3)
                            phase = 1 - phase;
                    }
                }
            } else
                phase = -1;
        }
        if (phase == -1) {
            data->value(dyn.active_l) << 1;
            data->value(dyn.active_r) << 1;
        } else if (phase == 0) {
            data->value(dyn.active_l) << 0;
            data->value(dyn.active_r) << 1;
        } else if (phase == 1) {
            data->value(dyn.active_l) << 1;
            data->value(dyn.active_r) << 0;
        }
        n++;
    });
    // propogate parameters
    graph.apply_all_binary_forward([&](node_data *cur, node_data *next) {
        cur->value(__y) = next->value(__x);
        for (auto &arg : cur->ocp_->expr_[__x]) {
            auto &next_arg = expr_index::get<sym>(arg->name_ + "_nxt");
            cur->value(next_arg) = next->value(arg);
        }
        next->value(dyn.active_l_cur) = cur->value(dyn.active_l);
        next->value(dyn.active_r_cur) = cur->value(dyn.active_r);
    });

    // std::cout << "\nleft\n";
    // graph.apply_all_unary_forward([&](node_data *data) {
    //     std::cout << data->value(dyn.active_l) << ',';
    // });
    // std::cout << "\nright\n";
    // graph.apply_all_unary_forward([&](node_data *data) {
    //     std::cout << data->value(dyn.active_r) << ',';
    // });
    // std::cout << "\n";

    solver.update(10);

    return 0;
}