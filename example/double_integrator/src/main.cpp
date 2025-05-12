#include <atri/core/directed_graph.hpp>
#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>

using namespace atri;

int main() {
    doubleIntegratorDyn dyn;
    doubleIntegratorCosts costs;
    auto prob = std::make_shared<problem>();
    prob->add(dyn);
    prob->add(costs.running(dyn.r, dyn.v, dyn.a));
    auto prob_terminal = prob->copy();
    prob_terminal->add(costs.terminal(dyn.r, dyn.v));

    utils::print_problem(prob);

    fmt::print("good!\n");

    ns_sqp sqp;

    auto init_node = sqp.graph_.add(ns_sqp::node(prob));
    auto end_node = sqp.graph_.add(ns_sqp::node(prob_terminal));

    sqp.graph_.add_edge(init_node, end_node, 100);

    sqp.graph_.set_head(init_node);
    sqp.graph_.set_tail(end_node);

    init_node->get(dyn.r).setConstant(6);

    sqp.update();

    fmt::print("nice!\n");

    return 0;
}