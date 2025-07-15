#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/impl/soft_constr.hpp>

namespace moto {
namespace ipm_impl {

struct ipm_data : public impl::soft_constr_data {
    vector g_; ///< ipm primal value
    vector r_s_; ///< ipm residuals g + t
    vector slack_;   ///< slack variables for the constraints
    vector diag_scaling; ///< Nesterov-Todd scaling T^{-1} N
    vector scaled_res_;  ///< residuals after NT scaling (Nr_g - r_s) T^{-1} = T{-1} N r_g + T^{-1} mu
    double mu_;          ///< barrier parameter
    vector d_slack_;     ///< newton step for slack variables
    vector d_multipler_; ///< newton step for multipliers
    ipm_data(sp_approx_map_ptr_t &&d)
        : impl::soft_constr_data(std::move(d)) {
        slack_.resize(func_.dim_);
        diag_scaling.resize(func_.dim_);
        scaled_res_.resize(func_.dim_);
    }
};

class ipm_constr : public impl::soft_constr {
  private:
    void value_impl(sp_approx_map &data) override final;
    void jacobian_impl(sp_approx_map &data) override final;

  public:
	using base = impl::soft_constr;
    using base::base;
    void initialize(data_type &data) override final;
    void post_rollout(data_type &data) override final;
    void line_search_step(data_type &data, solver::line_search_cfg *cfg) override final;
    void update_line_search_cfg(data_type &data, solver::line_search_cfg *cfg) override final;

    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return std::make_unique<ipm_data>(impl::soft_constr::make_approx_map(primal, raw, shared));
    }
};

} // namespace ipm_impl
using ipm = ipm_impl::ipm_constr;

} // namespace moto

#include <moto/ocp/constr.hpp>

#endif // MOTO_SOLVER_IPM_HPP