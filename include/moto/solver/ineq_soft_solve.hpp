#ifndef MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP
#define MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP

#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/data_base.hpp>

namespace moto {
namespace ineq_soft_solve {
void initialize(solver::data_base *data);
void post_rollout(solver::data_base *data);
void line_search_step(solver::data_base *data, workspace_data *config);
void calculate_line_search_bounds(solver::data_base *data, workspace_data *config);
template <typename Callback>
void for_each(solver::data_base *data, Callback &&callback) {
    data->for_each<soft_ineq_constr_fields>(
        [&](auto &f, auto &d) {
            auto &sf = dynamic_cast<impl::soft_constr &>(f);
            auto &sd = dynamic_cast<impl::soft_constr::soft_constr_data &>(d);
            callback(sf, sd);
        });
}
} // namespace ineq_soft_solve
} // namespace moto

#endif // MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP