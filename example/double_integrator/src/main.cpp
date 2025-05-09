#include <atri/ocp/core/shooting_node.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>

int main() {
    using namespace atri;
    doubleIntegratorDyn dyn;
    doubleIntegratorCosts costs;
    // auto cost = std::make_shared<doubleIntegratorCost>(dyn.r, dyn.v, dyn.a);
    auto prob = std::make_shared<problem>();
    prob->add(dyn);
    prob->add(costs.running(dyn.r, dyn.v, dyn.a));
    // prob->add(cost2);

    utils::print_problem(prob);

    auto &mem = data_mgr::get<node_data>();
    mem.create_data_batch(prob, 100);

    std::vector<shooting_node> nodes;
    for (int i = 0; i < 200; i++) {
        nodes.emplace_back(prob, mem);
    }

    // virtual terminal node?
    // copy problem -> add terminal cost
    auto prob_terminal = prob->copy();
    prob_terminal->add(costs.terminal(dyn.r_next, dyn.v_next));

    shooting_node terminal_node(prob_terminal, mem);
    nodes.back().swap(terminal_node);

    // set initial condition
    nodes.front().get(dyn.r).setOnes();

    for (size_t n = 0; n < nodes.size(); n++) {
        nodes[n].update_approximation();
    }

    fmt::print("good!\n");

    return 0;
}