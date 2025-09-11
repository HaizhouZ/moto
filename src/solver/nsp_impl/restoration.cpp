#include <moto/solver/ns_riccati/generic_solver.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
extern void print_debug(ns_riccati_data *cur);
void restoration() {
    // convert hard equality constraints to inequality constraints of form 0 <= c(x,u) <= 0
    // initialize the new multipliers to be 
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto