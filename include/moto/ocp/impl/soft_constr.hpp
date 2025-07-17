#ifndef MOTO_OCP_SOFT_CONSTR_HPP
#define MOTO_OCP_SOFT_CONSTR_HPP

#include <moto/ocp/impl/constr.hpp>

namespace moto {
namespace impl {
/**
 * @brief soft constraint data, contains additional primal step data for splitting post-rollout operation
 *
 */
struct soft_constr_approx_map : public constr_approx_map {
    std::vector<vector_ref> prim_step_;         // to be set
    std::vector<vector_ref> prim_corr_;         // to be set
    using constr_approx_map::constr_approx_map; ///< inherit constr_approx_map constructor
    soft_constr_approx_map(constr_approx_map &&rhs) : constr_approx_map(std::move(rhs)) {}
};

/**
 * @brief soft constraint interface class
 */
class soft_constr : public impl::soft_constr_base {
  protected:
    /// @brief basic data_type for the soft constraint. derived class can use this or make their own alias of @ref constr::constr_data
    using data_type = constr_data<soft_constr_approx_map, constr_approx_data>;

  public:
    using base = impl::soft_constr_base;
    using base::base; ///< inherit impl::constr constructor
    /// public type alias for @ref soft_constr_approx_map to ensure common interface of all soft constraints
    using sp_data_map = soft_constr_approx_map;

    /// initialize the soft constraint data
    virtual void initialize(sp_data_map &data) = 0;
    /// post rollout operation for the soft constraint to compute the newton step
    virtual void finalize_newton_step(sp_data_map &data) = 0;
    /// first order correction of the cost jacobian. vjp must be reset to zero before calling this
    virtual void correct_jacobian(sp_data_map &data) {};
    /// line search step for the soft constraint
    virtual void line_search_step(sp_data_map &data, workspace_data *worker_cfg) = 0;
    /// update the line search configuration (if necessary)
    virtual void update_linesearch_config(sp_data_map &data, workspace_data *worker_cfg) {}
    /***
     * @brief make approximation data for the soft constraint, will use default @ref data_type
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return sp_approx_map_ptr_t(make_approx<data_type>(primal, raw, shared));
    }
};
} // namespace impl
} // namespace moto

#endif // MOTO_OCP_SOFT_CONSTR_HPP