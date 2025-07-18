#ifndef MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP
#define MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP

#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/data_base.hpp>

namespace moto {
namespace solver {
namespace ineq_soft {
using soft_constr = impl::soft_constr;
using soft_constr_data_t = soft_constr::data_map_t;
template <typename Callback>
void for_each(data_base *data, Callback &&callback) {
    data->for_each<ineq_soft_constr_fields>(
        [&](auto &f, auto &d) {
            auto &sf = dynamic_cast<soft_constr &>(f);
            auto &sd = dynamic_cast<soft_constr_data_t &>(d);
            callback(sf, sd);
        });
}
template <typename Callback>
auto for_each(Callback &&callback) {
    return [callback = std::forward<Callback>(callback)](data_base *data) {
        for_each(data, callback);
    };
}
void initialize(data_base *data);
void finalize_newton_step(data_base *data);
void line_search_step(data_base *data, workspace_data *config);
void calculate_line_search_bounds(data_base *data, workspace_data *config);
/**
 * @brief prepare the first-order primal correction by calling to correct_jacobian on each soft constraint
 * @details will set the Q_x, Q_u, Q_y and prim_corr[__x] to zero and copy Q_y to Q_y_cache because 
 * it is @b assumed Q_y will be used in newton step finalization
 * @param data 
 */
void first_order_correction(data_base *data);
} // namespace ineq_soft
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP