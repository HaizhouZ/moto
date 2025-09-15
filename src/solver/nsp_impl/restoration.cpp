#include <moto/solver/ns_riccati/generic_solver.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
extern void print_debug(ns_riccati_data *cur);
void restoration() {
    // convert hard equality constraints to inequality constraints of form 0 <= c(x,u) <= 0
    // keeping original constraints, and create new ipm_constr for them, which is added to the problem as soft constraints
    // if it is detected that 
}
} // namespace ns_riccati
} // namespace solver
} // namespace moto