#ifndef MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP
#define MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP

#include <moto/ocp/impl/node_data.hpp>
#include <moto/ocp/soft_constr.hpp>
#include <moto/solver/data_base.hpp>

namespace moto {
class node_data; // forward declaration
namespace solver {
namespace ineq_soft {
using soft_constr = soft_constr;
using soft_constr_data_t = soft_constr::data_map_t;
/// @brief Get a lambda that applies the callback to each soft constraint
/// @tparam Callback
/// @param callback
/// @return
template <typename Callback>
    requires std::is_invocable_v<Callback, const soft_constr &, soft_constr_data_t &>
auto get_for_each(Callback &&callback) {
    return [callback = std::forward<Callback>(callback)]<typename data_type>(data_type *data) -> void {
        dynamic_cast<node_data *>(data)->for_each<ineq_soft_constr_fields>(
            [&](const soft_constr &sf, soft_constr_data_t &sd) {
            callback(sf, sd);
        });
    };
}
/// @brief for each soft constraint, call the callback with (const soft_constr&, soft_constr::data_map_t&)
inline void for_each(auto *data, auto &&callback) {
    get_for_each(std::forward<decltype(callback)>(callback))(data);
}
void bind_runtime(const soft_constr &sf, soft_constr_data_t &sd);
void bind_runtime(node_data *data);
void ensure_initialized(const soft_constr &sf, soft_constr_data_t &sd);
/**
 * @brief finalize the newton step for the soft constraints
 * @details it will call finalize_newton_step on each soft constraint
 * @param data data base
 */
void finalize_newton_step(node_data *data);
/**
 * @brief finalize the predictor step, should be called after the rollout (@ref finalize_newton_step)
 * @details it will call finalize_predictor_step on each soft constraint
 * @param data data base
 * @param config workspace data pointer to the config to be updated
 */
void finalize_predictor_step(node_data *data, workspace_data *config);
/**
 * @brief line search step for the soft constraints
 * @details it will call apply_affine_step on each soft constraint
 * @param data data base
 * @param config workspace data pointer (should contain linesearch config) to the config to be used
 */
void apply_affine_step(node_data *data, workspace_data *config);
/**
 * @brief calculate the line search bounds for the soft constraints
 * @details it will call update_ls_bounds on each soft constraint
 * @param data data base
 * @param config workspace data pointer to the config to be updated
 */
void update_ls_bounds(node_data *data, workspace_data *config);
/**
 * @brief back up the current soft-constraint trial state
 * @param data data base
 */
void backup_trial_state(node_data *data);
/**
 * @brief restore the backed-up soft-constraint trial state
 * @param data data base
 */
void restore_trial_state(node_data *data);
/**
 * @brief prepare for the first-order primal correction and call to apply_corrector_step on each soft constraint
 * @details it will set prim_corr[__x] to zero and swap the active stage gradient with the
 * lagrangian-gradient correction buffer (as a pre-correction cache),
 * i.e., later solving will use the correction right-hand side
 * it is @b assumed Q_y will later be used in newton step finalization
 * @param data
 */
void corrector_step_start(data_base *data);
/**
 * @brief finalize the first-order primal correction and restore the active stage
 * gradient after correction
 * @details it will swap back the active stage gradient and lagrangian-gradient
 * correction buffer, then fold the local y correction into Q_y
 * @param data
 */
void corrector_step_end(data_base *data);
} // namespace ineq_soft
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_INEQ_SOFT_SOLVE_HPP
