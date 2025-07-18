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
class ipm_config {
  private:
    scalar_t sig = 1.0;                   ///< centering parameter
    bool ipm_compute_affine_step = false; ///< whether in affine step computation
    bool ipm_reject_corrector = false;    ///< whether to reject the corrector step
  public:
    scalar_t mu = 1e-2;                                     ///< initial barrier parameter
    adaptive_mu_t mu_method = mehrotra_predictor_corrector; ///< adaptive mu method
    bool adaptive_mu_allowed = false;                       ///< whether to adapt mu during line search
    bool ipm_conditional_corrector = false;                 ///< whether to use conditional corrector
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
    bool ipm_accept_corrector() {
        return ipm_enable_corrector() && !ipm_reject_corrector;
    }
    bool ipm_computing_affine_step() {
        return ipm_enable_affine_step() && ipm_compute_affine_step;
    }
    void ipm_start_predictor_computation() {
        if (ipm_enable_affine_step()) {
            ipm_compute_affine_step = true;
        }
    }
    void ipm_end_predictor_computation() {
        if (ipm_enable_affine_step()) {
            ipm_compute_affine_step = false;
        }
    }

    void adaptive_mu_update(worker &ipm_worker);
};
} // namespace ipm_impl
} // namespace moto

#endif // MOTO_SOLVER_ipm_config_HPP