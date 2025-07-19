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
/**
 * @brief finalize the newton step for the soft constraints
 * @details it will call finalize_newton_step on each soft constraint
 * @param data data base
 */
void finalize_newton_step(data_base *data);
/**
 * @brief finalize the predictor step, should be called after the rollout (@ref finalize_newton_step)
 * @details it will call finalize_predictor_step on each soft constraint
 * @param data data base
 * @param config workspace data pointer to the config to be updated
 */
void finalize_predictor_step(data_base *data, workspace_data *config);
/**
 * @brief line search step for the soft constraints
 * @details it will call line_search_step on each soft constraint
 * @param data data base
 * @param config workspace data pointer (should contain linesearch config) to the config to be used
 */
void line_search_step(data_base *data, workspace_data *config);
/**
 * @brief calculate the line search bounds for the soft constraints
 * @details it will call update_linesearch_config on each soft constraint
 * @param data data base
 * @param config workspace data pointer to the config to be updated
 */
void calculate_line_search_bounds(data_base *data, workspace_data *config);
/**
 * @brief prepare for the first-order primal correction and call to correct_jacobian on each soft constraint
 * @details it will set prim_corr[__x] to zero and swap merit jacobian and its modifcation (as a pre-correction cache),
 * i.e., later solving will use the jacobian modification
 * it is @b assumed Q_y will be used in newton step finalization
 * @param data
 */
void first_order_correction_start(data_base *data);
/**
 * @brief finalize the first-order primal correction and cache Q_y after correction
 * @details it will swap back merit jacobian and its modification and set Q_y_corr to the Q_y correction
 * it is @b assumed Q_y will be used in newton step finalization
 * @param data
 */
void first_order_correction_end(data_base *data);
} // namespace ineq_soft
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP