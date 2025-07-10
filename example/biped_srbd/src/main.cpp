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
    prob->add(dyn.active_l);
    prob->add(dyn.active_r);
    prob->add(dyn.active_l_cur);
    prob->add(dyn.active_r_cur);
    // prob->add(dyn.friction_cone());
    prob->add(dyn.running_cost());
    // prob->add(dyn.stance_foot_constr());
    // prob->add(dyn.foot_loc_constr());
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
    init_node->value(dyn.r_d) << 0., 0., 0.; // desired position of the com

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
        // if (phase == -1) {
        data->value(dyn.active_l) << 1;
        data->value(dyn.active_r) << 1;
        // } else if (phase == 0) {
        //     data->value(dyn.active_l) << 0;
        //     data->value(dyn.active_r) << 1;
        // } else if (phase == 1) {
        //     data->value(dyn.active_l) << 1;
        //     data->value(dyn.active_r) << 0;
        // }
        n++;
    });
    // propogate parameters
    graph.apply_all_binary_forward([&](node_data *cur, node_data *next) {
        dynamics::copy_x_to_y(next->value(__x), cur->value(__y), next->ocp_, cur->ocp_);
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
    solver.update(1);
    size_t step = 0;
    graph.apply_all_unary_forward([&](node_data *data) {
        std::cout << "------------- Step: " << step++ << '\n';
        std::cout << "cost: " << data->cost() << '\n';
        std::cout << "dyn_res: " << data->value(__dyn).transpose() << '\n';
        std::cout << "dual: " << data->dense_->dual_[__dyn].transpose() << '\n';
        // std::cout << "dynjacx: \n" << data->dense_->approx_[__dyn].jac_[__x] << '\n';
        // std::cout << "dynjacu: \n" << data->dense_->approx_[__dyn].jac_[__u] << '\n';
        // std::cout << "dynjacy: \n" << data->dense_->approx_[__dyn].jac_[__y] << '\n';
        // std::cout << "costx: " << data->dense_->jac_[__x] << '\n';
        // std::cout << "costu: " << data->dense_->jac_[__u] << '\n';
        // std::cout << "costy: " << data->dense_->jac_[__y] << '\n';
        // std::cout << "costxx: \n" << data->dense_->hessian_[__x][__x] << '\n';
        // std::cout << "costxu: \n" << data->dense_->hessian_[__u][__x] << '\n';
        // std::cout << "costuu: \n" << data->dense_->hessian_[__u][__u] << '\n';
        // std::cout << "costyy: \n" << data->dense_->hessian_[__y][__y] << '\n';
        // std::cout << "costyx: \n" << data->dense_->hessian_[__y][__x] << '\n';
        // std::cout << "costyu: \n" << data->dense_->hessian_[__y][__u] << '\n';
        std::cout << "r: " << data->value(dyn.r).transpose() << '\n';
        std::cout << "r_n: " << data->value(dyn.r_n).transpose() << '\n';
        std::cout << "v: " << data->value(dyn.v).transpose() << '\n';
        std::cout << "v_n: " << data->value(dyn.v_n).transpose() << '\n';
        std::cout << "r_d: " << data->value(dyn.r_d).transpose() << '\n';
        std::cout << "r_l: " << data->value(dyn.r_l).transpose() << '\n';
        std::cout << "r_l_n: " << data->value(dyn.r_l_n).transpose() << '\n';
        std::cout << "r_r: " << data->value(dyn.r_r).transpose() << '\n';
        std::cout << "r_r_n: " << data->value(dyn.r_r_n).transpose() << '\n';
        std::cout << "v_l: " << data->value(dyn.v_l).transpose() << '\n';
        std::cout << "v_r: " << data->value(dyn.v_r).transpose() << '\n';
        std::cout << "v_d: " << data->value(dyn.v_n).transpose() << '\n';
        std::cout << "f_l: " << data->value(dyn.f_l).transpose() << '\n';
        std::cout << "f_r: " << data->value(dyn.f_r).transpose() << '\n';
    });

    return 0;
}