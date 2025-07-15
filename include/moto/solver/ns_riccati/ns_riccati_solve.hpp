#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {

namespace solver {
struct line_search_cfg;
}
namespace ns_riccati {

void update_approx(ns_node_data *cur);
void ns_factorization(ns_node_data *cur);
void partial_value_derivative(ns_node_data *prev, ns_node_data *cur);
void riccati_recursion(ns_node_data *cur, ns_node_data *prev);
void compute_primal_sensitivity(ns_node_data *cur);
void fwd_linear_rollout(ns_node_data *cur, ns_node_data *next);
void finalize_newton_step(ns_node_data *cur);
void line_search_step(ns_node_data *cur, solver::line_search_cfg *cfg);
} // namespace ns_riccati

} // namespace moto

#endif