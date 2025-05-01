#include <atri/ocp/core/shooting_node.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/dynamics.hpp>

int main() {
    using namespace atri;
    doubleIntegratorDyn dyn;
    auto prob = std::make_shared<problem>();
    prob->add(dyn);

    utils::print_problem(prob);

    auto &mem = data_mgr::get<node_data>();
    mem.create_data_batch(prob, 100);

    std::vector<shooting_node> nodes;
    for (int i = 0; i < 200; i++) {
        nodes.emplace_back(prob, mem);
    }

    for (auto &n : nodes) {
        n.update_approximation();
    }

    fmt::print("good!\n");

    return 0;
}