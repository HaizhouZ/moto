#include <moto/solver/ipm_constr.hpp>

namespace moto {
namespace ipm {
void ipn_constr_impl::value_impl(sp_approx_map &data) {
    constr_impl::value_impl(data);
}
void ipn_constr_impl::jacobian_impl(sp_approx_map &data) {
    constr_impl::jacobian_impl(data);
    auto &d = static_cast<ipm_data &>(data);
    // setup T^{-1} N
    d.diag_scaling.noalias() = d.multiplier_.cwiseQuotient(d.slack_);
    // set scaled residual
    d.scaled_residual = d.diag_scaling.cwiseProduct(d.v_);
    d.scaled_residual.array() += d.mu_ / d.slack_.array();
    // modification of jacobian
    size_t j_idx = 0;
    for (auto &j : d.jac_) {
        if (j.rows() != 0 && j.cols() != 0) { // skip empty jac
            d.vjp_[j_idx].noalias() += d.scaled_residual.asDiagonal() * j;
        }
        j_idx++;
    }
    // modification of hessian
    size_t outer_idx = 0;
    for (auto &outer : d.hess_) {
        size_t inner_idx = 0;
        if (outer.size()) { // skip empty hess
            for (auto &inner : outer) {
                if (inner.rows() != 0 && inner.cols() != 0) {
                    inner.noalias() += d.jac_[outer_idx].transpose() * d.diag_scaling.asDiagonal() * d.jac_[inner_idx];
                }
                inner_idx++;
            }
        }
        outer_idx++;
    }
}
} // namespace ipm
} // namespace moto