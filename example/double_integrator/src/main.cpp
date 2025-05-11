#include <atri/core/directed_graph.hpp>
#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/ns_sqp.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/cost.hpp>
#include <double_integrator/dynamics.hpp>

using namespace atri;

// struct sym : public expr_ptr_t {
//     sym(const std::string &name, size_t dim, field_t type)
//         : expr_ptr_t(new expr(name, dim, type)) {
//         assert(size_t(type) <= field::num_sym);
//     }
// };

int main() {
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

    // should seperate shooting nodes from the solver?
    // creating shooting nodes requires access mem
    // mem should not be managed by user!
    // nodeSets.directed_view()
    // nodeSets.contiguous_view()
    directed_graph<shooting_node> graph;
    auto init_node = graph.add(shooting_node(prob, mem));
    auto end_node = graph.add(shooting_node(prob_terminal, mem));
    auto end_node2 = graph.add(shooting_node(prob_terminal, mem));
    graph.add_edge(init_node, end_node, 100);
    graph.add_edge(init_node, end_node2, 60);
    graph.add_edge(end_node2, end_node, 70);
    // iterate on edge
    // for n in edge.nodes:...
    // then edge.st
    // then
    graph.flatten();
    graph.set_head(init_node);
    graph.set_tail(end_node);
    std::atomic<size_t> cnt = 0;

    graph.apply_all_binary_forward([&cnt](shooting_node& a, shooting_node& b) { cnt++; });
    fmt::print("forward {}\n", cnt.load());
    cnt = 0;
    graph.apply_all_binary_backward([&cnt](shooting_node& a, shooting_node& b) { cnt++; });
    fmt::print("backward {}\n", cnt.load());

    fmt::print("nice!\n");

    return 0;
}