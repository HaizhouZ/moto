#ifndef MOTO_SOLVER_ipm_config_HPP
#define MOTO_SOLVER_ipm_config_HPP

#include <moto/core/fwd.hpp>
#include <new>

namespace moto {
namespace ipm_impl {
enum adaptive_mu_t : size_t {
    mehrotra_predictor_corrector = 0, ///< Mehrotra predictor-corrector method
    mehrotra_probing,                 ///< Mehrotra predictor method
    quality_function_based,           ///< quality function based method
};
struct ipm_config {
    scalar_t mu = 1e-2;                                     ///< initial barrier parameter
    scalar_t sig = 1.0;                                     ///< centering parameter
    adaptive_mu_t mu_method = mehrotra_predictor_corrector; ///< adaptive mu method
    bool adaptive_mu = false;                               ///< whether to adapt mu during line search
    struct alignas(std::hardware_destructive_interference_size) worker {
        size_t n_ipm_cstr = 0;
        scalar_t prev_normalized_comp = 0.;  ///< previous normalized complementarity
        scalar_t after_normalized_comp = 0.; ///< after line search normalized complementarity
    };
    using worker_type = worker;
    bool comp_affine_step() {
        return mu_method == mehrotra_predictor_corrector || mu_method == mehrotra_probing;
    }
};
} // namespace ipm_impl
} // namespace moto

#endif // MOTO_SOLVER_ipm_config_HPP