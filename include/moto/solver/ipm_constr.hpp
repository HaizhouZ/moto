#ifndef MOTO_SOLVER_IPM_CONSTR_HPP
#define MOTO_SOLVER_IPM_CONSTR_HPP

#include <moto/ocp/constr.hpp>

namespace moto {
namespace ipm {

struct ipm_data : public constr_data {
    vector slack_;          ///< slack variables for the constraints
    vector diag_scaling;    ///< Nesterov-Todd scaling T^{-1} N
    vector scaled_residual; ///< residuals after NT scaling (Nr_g - r_s) T^{-1} = T{-1} N r_g + T^{-1} mu
    double mu_;             ///< barrier parameter
    vector d_slack_;        // newton step for slack variables
    vector d_multipler_;    // newton step for multipliers
    ipm_data(approx_storage &raw, constr_data &&d, constr_impl *f)
        : constr_data(raw, std::move(d), f) {
        slack_.resize(f->dim_);
        diag_scaling.resize(f->dim_);
        scaled_residual.resize(f->dim_);
    }
};

class ipn_constr_impl : public constr_impl {
  private:
    void value_impl(sp_approx_map &data) override final;
    void jacobian_impl(sp_approx_map &data) override final;

  public:
    using constr_impl::constr_impl;

    /**
     * @brief make the sparse approximation data for the IPM
     * @param primal sym data including states inputs etc
     * @param raw dense raw data of approximation
     * @param shared shared data
     * @return sp_approx_map_ptr_t
     */
    sp_approx_map_ptr_t make_approx_map(sym_data &primal, approx_storage &raw, shared_data &shared) override {
        return sp_approx_map_ptr_t(new ipm_data(raw, dynamic_cast<constr_data &&>(*constr_impl::make_approx_map(primal, raw, shared)), this));
    }
};

} // namespace ipm
} // namespace moto

#endif // MOTO_SOLVER_IPM_HPP