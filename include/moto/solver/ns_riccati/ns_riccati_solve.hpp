#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
/**
 * @brief update the approximation of the node data
 * @note will start from reset Q-derivatives to zero
 * @param cur current node data
 */
void update_approx(ns_node_data *cur);
/**
 * @brief factorization for the nullspace kernel using the hard constraints
 * @details will directly add related parts to the Q-derivatives
 * @param cur current node data
 */
void ns_factorization(ns_node_data *cur);
/**
 * @brief perform the riccati recursion for the current node
 * @details will update the Q-derivatives and the nullspace data
 * @param cur current node data
 * @param prev previous node data, can be nullptr
 */
void riccati_recursion(ns_node_data *cur, ns_node_data *prev);
/**
 * @brief perform the riccati recursion correction for the current node
 * @details will update the first-order Q-derivatives and the nullspace residuals
 * @param cur current node data
 * @param prev previous node data, can be nullptr
 */
void riccati_recursion_correction(ns_node_data *cur, ns_node_data *prev);
/**
 * @brief compute the primal sensitivity for the current node
 * @details will update the d_u and d_y sensitivity
 * @param cur current node data
 */
void compute_primal_sensitivity(ns_node_data *cur);
/**
 * @brief compute the primal sensitivity correction for the current node
 * @details will update the d_u and d_y sensitivity using the nullspace residual correction
 * @param cur current node data
 */
void compute_primal_sensitivity_correction(ns_node_data *cur);
/**
 * @brief perform the forward linear rollout for the current node
 *
 * @param cur current node data
 * @param next next node data, can be nullptr
 */
void fwd_linear_rollout(ns_node_data *cur, ns_node_data *next);
/**
 * @brief perform the forward linear rollout correction for the current node
 * @details will update the prim_corr[__x] and prim_corr[__y]
 * @param cur current node data
 * @param next next node data, can be nullptr
 */
void fwd_linear_rollout_correction(ns_node_data *cur, ns_node_data *next);
/**
 * @brief finalize the newton step for the current node
 * @details will update the d_lbd_f and d_lbd_s_c
 * @param cur current node data
 * @param finalize_dual whether to finalize the dual step
 */
void finalize_newton_step(ns_node_data *cur, bool finalize_dual = true);
/**
 * @brief finalize the newton step for the current node after correction
 * @details will correct the primal steps before finalizing the dual steps
 * @param cur current node data
 */
void finalize_newton_step_correction(ns_node_data *cur);
/**
 * @brief line search step for the current node
 *
 * @param cur current node data
 * @param cfg workspace configuration containing line search parameters
 */
void line_search_step(ns_node_data *cur, workspace_data *cfg);
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif