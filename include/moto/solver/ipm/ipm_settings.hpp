#ifndef MOTO_SOLVER_IPM_SETTINGS_HPP
#define MOTO_SOLVER_IPM_SETTINGS_HPP

#include <moto/core/fwd.hpp>

namespace moto {
namespace ipm_impl {
enum adaptive_mu_t : size_t {
    mehrotra_predictor_corrector = 0, ///< Mehrotra predictor-corrector method
    mehrotra_probing,                 ///< Mehrotra predictor method
    quality_function_based,           ///< quality function based method
};
struct ipm_settings {
    scalar_t mu = 1e-2;                                      ///< initial barrier parameter
    scalar_t sig = 1.0;                                     ///< centering parameter
    adaptive_mu_t mu_method = mehrotra_predictor_corrector; ///< adaptive mu method
};
} // namespace ipm_impl
} // namespace moto

#endif // MOTO_SOLVER_IPM_SETTINGS_HPP