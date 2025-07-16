#ifndef MOTO_SOLVER_IPM_ADAPTIVE_MU_HPP
#define MOTO_SOLVER_IPM_ADAPTIVE_MU_HPP

#include <moto/solver/ipm/ipm_constr.hpp>

namespace moto {
namespace ipm_impl {
/// @brief adaptive mu strategy for all IPM constraints in a problem
// void adaptive_mu(std::vector<ipm_constr *> &constraints, scalar_t &mu, scalar_t &sig,
//                  ipm_approx_data::adaptive_mu method = ipm_approx_data::mehrotra_predictor_corrector) {
//     for (auto &c : constraints) {
//         c->data<ipm_approx_data>().mu_method = method;
//         c->data<ipm_approx_data>().mu_ = mu;
//         c->data<ipm_approx_data>().sig_ = sig;
//     }
// }
} // namespace ipm_impl
} // namespace moto

#endif // MOTO_SOLVER_IPM_ADAPTIVE_MU_HPP