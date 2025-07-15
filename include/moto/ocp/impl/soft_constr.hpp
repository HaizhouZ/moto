#ifndef MOTO_OCP_SOFT_CONSTR_HPP
#define MOTO_OCP_SOFT_CONSTR_HPP

#include <moto/ocp/impl/constr.hpp>

namespace moto {
namespace solver {
struct line_search_cfg;
}
namespace impl {
/**
 * @brief soft constraint data, contains additional primal step data for splitting post-rollout operation
 *
 */
struct soft_constr_data : public constr_data {
    std::vector<vector_ref> prim_step_; // to be set
    soft_constr_data(sp_approx_map_ptr_t &&d)
        : constr_data(std::move(d)) {
    }
};
/**
 * @brief soft constraint interface class
 * @note must implement make_approx_map
 */
class soft_constr : public impl::soft_constr_base {
  public:
    using base = impl::soft_constr_base;
    using base::base; // inherit impl::constr constructor
    using data_type = soft_constr_data;

    virtual void initialize(soft_constr_data &data) = 0;
    virtual void post_rollout(soft_constr_data &data) = 0;
    virtual void line_search_step(soft_constr_data &data, solver::line_search_cfg *cfg) = 0;
    virtual void update_line_search_cfg(soft_constr_data &data, solver::line_search_cfg *cfg) {}
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return std::make_unique<soft_constr_data>(base::make_approx_map(primal, raw, shared));
    };
};
} // namespace impl
} // namespace moto

#endif // MOTO_OCP_SOFT_CONSTR_HPP