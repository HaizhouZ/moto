#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/impl/soft_constr.hpp>
#include <moto/solver/ipm/ipm_settings.hpp>

namespace moto {
namespace ipm_impl {

struct ipm_approx_data : public impl::constr_approx_data {
    ipm_settings *ipm_cfg = nullptr; ///< pointer to the IPM settings

    vector g_;           ///< ipm primal value
    vector r_s_;         ///< ipm residuals g + t
    vector slack_;       ///< slack variables for the constraints
    vector diag_scaling; ///< Nesterov-Todd scaling T^{-1} N
    vector scaled_res_;  ///< residuals after NT scaling (Nr_g - r_s) T^{-1} = T{-1} N r_g + T^{-1} mu
    vector d_slack_;     ///< newton step for slack variables
    vector d_multipler_; ///< newton step for multipliers
    ipm_approx_data(constr_approx_data &&rhs)
        : impl::constr_approx_data(std::move(rhs)) {
        slack_.resize(f_->dim_);
        diag_scaling.resize(f_->dim_);
        scaled_res_.resize(f_->dim_);
    }
};

class ipm_constr final : public impl::soft_constr {
  private:
    using base = impl::soft_constr;
    /// + update the IPM slack and residuals
    void value_impl(sp_approx_map &data) override final;
    /// + update the IPM-modified cost jacobian and hessian
    void jacobian_impl(sp_approx_map &data) override final;
    /// data type for the IPM constraint
    using data_type = constr_data<base::data_type::mtype, ipm_approx_data>;

  public:
    using base::base;
    void setup_solver_setting(sp_approx_map &data, solver::solver_settings *settings) override {
        base::setup_solver_setting(data, settings);
        auto &d = dynamic_cast<ipm_approx_data &>(data);
        d.ipm_cfg = dynamic_cast<ipm_settings *>(settings);
    }
    /// @brief initialize the IPM constraint data
    void initialize(soft_constr_data &data) override final;
    /// @brief post rollout operation for the IPM constraint to compute the newton step
    void post_rollout(soft_constr_data &data) override final;
    /// @brief line search step for the IPM constraint
    void line_search_step(soft_constr_data &data, solver::line_search_cfg *cfg) override final;
    /// @brief update the line search configuration (if necessary)
    void update_line_search_cfg(soft_constr_data &data, solver::line_search_cfg *cfg) override final;

    using ipm_data = data_type;

    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return sp_approx_map_ptr_t(make_approx<data_type>(primal, raw, shared));
    }
};
} // namespace ipm_impl
using ipm = ipm_impl::ipm_constr;

} // namespace moto

#include <moto/ocp/constr.hpp>

#endif // MOTO_SOLVER_IPM_HPP