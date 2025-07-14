#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <moto/solver/ns_riccati_data.hpp>

namespace moto {

namespace solver {
struct line_search_cfg;
}
namespace nullsp_kkt_solve {

void update_approx(riccati_data *cur);
void ns_factorization(riccati_data *cur);
void partial_value_derivative(riccati_data *prev, riccati_data *cur);
void riccati_recursion(riccati_data *cur, riccati_data *prev);
void compute_primal_sensitivity(riccati_data *cur);
void fwd_linear_rollout(riccati_data *cur, riccati_data *next);
void finalize_newton_step(riccati_data *cur);
void line_search_step(riccati_data *cur, solver::line_search_cfg* cfg);
} // namespace nullsp_kkt_solve

} // namespace moto

#endif