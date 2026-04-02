#ifndef MOTO_SOLVER_NS_RICCATI_SOLVER_HPP
#define MOTO_SOLVER_NS_RICCATI_SOLVER_HPP

#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {
struct workspace_data;
struct node_data;
namespace solver {
namespace ns_riccati {
struct generic_solver {

    virtual ns_riccati_data create_data(node_data *full_data);
    /**
     * @brief factorization for the nullspace kernel using the hard constraints.
     * @details will directly add related parts to the Q-derivatives.
     *          When @p gauss_newton is true, instead of projecting the equality constraints
     *          into the cost via nullspace, treats them as objectives (Gauss-Newton mode):
     *          adds GN gradient/Hessian terms from dynamics + equality residuals + proximal
     *          anchors (read from cur->aux_ as restoration_aux_data), and runs the Riccati
     *          in unconstrained mode while preserving lu_eq_ for the dual step.
     * @param cur          current node data
     * @param gauss_newton if true, run in restoration/Gauss-Newton mode
     */
    virtual void ns_factorization(ns_riccati_data *cur, bool gauss_newton = false);
    /**
     * @brief factorization for the correction step
     * @details reuses the standard nullspace factorization without merging Jacobian modifications
     * @param cur current node data
     */
    virtual void ns_factorization_correction(ns_riccati_data *cur);
    /**
     * @brief perform the riccati recursion for the current node
     * @details will update the Q-derivatives and the nullspace data
     * @param cur current node data
     * @param prev previous node data, can be nullptr
     */
    virtual void riccati_recursion(ns_riccati_data *cur, ns_riccati_data *prev);
    /**
     * @brief perform the riccati recursion correction for the current node
     * @details will update the first-order Q-derivatives and the nullspace residuals
     * @param cur current node data
     * @param prev previous node data, can be nullptr
     */
    virtual void riccati_recursion_correction(ns_riccati_data *cur, ns_riccati_data *prev);
    /**
     * @brief compute the primal sensitivity for the current node
     * @details will update the d_u and d_y sensitivity
     * @param cur current node data
     */
    virtual void compute_primal_sensitivity(ns_riccati_data *cur);
    /**
     * @brief compute the primal sensitivity correction for the current node
     * @details will update the d_u and d_y sensitivity using the nullspace residual correction
     * @param cur current node data
     */
    virtual void compute_primal_sensitivity_correction(ns_riccati_data *cur);
    /**
     * @brief perform the forward linear rollout for the current node
     *
     * @param cur current node data
     * @param next next node data, can be nullptr
     */
    virtual void fwd_linear_rollout(ns_riccati_data *cur, ns_riccati_data *next);
    /**
     * @brief perform the forward linear rollout correction for the current node
     * @details will update the prim_corr[__x] and prim_corr[__y]
     * @param cur current node data
     * @param next next node data, can be nullptr
     */
    virtual void fwd_linear_rollout_correction(ns_riccati_data *cur, ns_riccati_data *next);
    /**
     * @brief finalize the newton step for the current node
     * @details will update the d_lbd_f and d_lbd_s_c
     * @param cur current node data
     * @param finalize_dual whether to finalize the dual step
     */
    virtual void finalize_primal_step(ns_riccati_data *cur);

    virtual void finalize_dual_newton_step(ns_riccati_data *cur);
    /**
     * @brief finalize the newton step for the current node after correction
     * @note will update the Q derivatives with the correction terms, so it should be called after the correction step
     * @param cur current node data
     */
    virtual void finalize_primal_step_correction(ns_riccati_data *cur);
    /**
     * @brief line search step for the current node
     *
     * @param cur current node data
     * @param cfg workspace configuration containing line search parameters
     */
    virtual void apply_affine_step(ns_riccati_data *cur, workspace_data *cfg);

    void compute_kkt_residual(ns_riccati_data *cur);
};
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif
