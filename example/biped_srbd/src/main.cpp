#include <dynamics.hpp>
#include <iostream>
#include <moto/ocp/problem.hpp>

int main() {
    biped_srbd::srbd_dynamics dyn;
    std::cout << "Hello, Biped SRBD!" << std::endl;
    auto prob = moto::ocp::make();
    prob->add(dyn.euler());
    prob->add(dyn.friction_cone());
    prob->add(dyn.running_cost());
    prob->add(dyn.terminal_cost());
    prob->add(dyn.stance_foot_constr());
    prob->add(dyn.foot_loc_constr());
    return 0;
}