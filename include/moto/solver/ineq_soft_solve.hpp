#ifndef MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP
#define MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP

#include <moto/solver/data_base.hpp>

namespace moto {
namespace ineq_soft_solve {

void initialize(solver::data_base *data);
void post_rollout(solver::data_base *data);
void line_search_step(solver::data_base *cur, scalar_t alpha);
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP