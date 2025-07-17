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
    bool adaptive_mu_allowed = false;                       ///< whether to adapt mu during line search
    bool ipm_compute_affine_step = false;                   ///< whether in affine step computation
    struct alignas(std::hardware_destructive_interference_size) worker {
        size_t n_ipm_cstr = 0;
        scalar_t prev_aff_comp = 0.; ///< previous complementarity without the affine step
        scalar_t post_aff_comp = 0.; ///< complementarity after adding the affine step
    };
    using worker_type = worker;
    bool ipm_enable_affine_step() {
        return adaptive_mu_allowed && (mu_method == mehrotra_predictor_corrector || mu_method == mehrotra_probing);
    }
    bool ipm_enable_corrector() {
        return adaptive_mu_allowed && mu_method == mehrotra_predictor_corrector;
    }

    void adaptive_mu_update(worker &ipm_worker);
};
} // namespace ipm_impl
} // namespace moto

#endif // MOTO_SOLVER_ipm_config_HPP