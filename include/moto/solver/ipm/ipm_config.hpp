#ifndef MOTO_SOLVER_ipm_config_HPP
#define MOTO_SOLVER_ipm_config_HPP

#include <moto/core/fwd.hpp>
#include <new>

namespace moto {
namespace solver {
class ipm_config {
  private:
    scalar_t sig = 1.0;                   ///< centering parameter
    bool ipm_compute_affine_step = false; ///< whether in affine step computation
    bool ipm_reject_corrector = false;    ///< whether to reject the corrector step
  public:
    enum adaptive_mu_t : size_t {
        mehrotra_predictor_corrector = 0, ///< Mehrotra predictor-corrector method
        mehrotra_probing,                 ///< Mehrotra predictor method
        quality_function_based,           ///< quality function based method
        monotonic_decrease,               ///< monotonic decrease method
    };

    scalar_t mu = 1e-2; ///< initial barrier parameter
    adaptive_mu_t mu_method = mehrotra_predictor_corrector; ///< adaptive mu method
    bool ipm_conditional_corrector = false; ///< whether to use conditional corrector

    struct MOTO_ALIGN_NO_SHARING worker {
        scalar_t kkt_inf = std::numeric_limits<scalar_t>::infinity(); ///< KKT infinity norm, used for logging and debugging
        size_t n_ipm_cstr = 0;
        scalar_t prev_aff_comp = 0.; ///< previous complementarity without the affine step
        scalar_t post_aff_comp = 0.; ///< complementarity after adding the affine step
        worker &operator+=(const worker &rhs) {
            if (&rhs != this) {
                n_ipm_cstr += rhs.n_ipm_cstr;
                prev_aff_comp += rhs.prev_aff_comp;
                post_aff_comp += rhs.post_aff_comp;
            }
            return *this;
        }
    };
    worker worker_data;
    using worker_type = worker;
    bool is_adaptive_mu() const {
        return mu_method == mehrotra_predictor_corrector || mu_method == mehrotra_probing;
    }
    bool ipm_enable_affine_step() const {
        return is_adaptive_mu() && (mu_method == mehrotra_predictor_corrector || mu_method == mehrotra_probing);
    }
    bool ipm_enable_corrector() const {
        return is_adaptive_mu() && mu_method == mehrotra_predictor_corrector;
    }
    bool ipm_accept_corrector() const {
        return ipm_enable_corrector() && !ipm_reject_corrector;
    }
    bool ipm_computing_affine_step() const {
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
} // namespace solver
} // namespace moto

#endif // MOTO_SOLVER_ipm_config_HPP