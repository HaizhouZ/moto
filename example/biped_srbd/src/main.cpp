#include <iostream>
#include <dynamics.hpp>
#include <moto/ocp/problem.hpp>

int main() {
    biped_srbd::srbd_dynamics dyn;
    auto euler_constr = dyn.euler();
    std::cout << "Hello, Biped SRBD!" << std::endl;
    auto prob = moto::ocp::make();
    prob->add(euler_constr);
    return 0;
}