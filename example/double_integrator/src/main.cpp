#include <atri/ocp/core/shooting_node.hpp>
#include <atri/utils/print.hpp>
#include <double_integrator/dynamics.hpp>

int main() {
    using namespace atri;
    auto dyn = std::make_shared<doubleIntegratorDyn>();
    auto prob = std::make_shared<problem>();
    prob->add(dyn);
    prob->add(dyn->in_args());

    utils::print_problem(prob);

    auto &mem = data_mgr::get<node_data>();

    return 0;
}