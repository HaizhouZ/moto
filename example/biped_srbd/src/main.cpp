#include <dynamics.hpp>
#include <iostream>
#include <moto/ocp/problem.hpp>
#include <moto/solver/ns_sqp.hpp>

int main() {
    moto::func_codegen_helper::enable();
    biped_srbd::srbd_dynamics dyn;
    std::cout << "Hello, Biped SRBD!" << std::endl;
    auto prob = moto::ocp::make();
    prob->add(dyn.euler());
    prob->add(dyn.friction_cone());
    prob->add(dyn.running_cost());
    prob->add(dyn.stance_foot_constr());
    prob->add(dyn.foot_loc_constr());
    prob->add(dyn.terminal_cost());
    moto::func_codegen_helper::wait_until_all_compiled(8);
    return 0;
}