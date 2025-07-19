#include <moto/solver/ipm/ipm_config.hpp>

namespace moto {
namespace solver {
void ipm_config::adaptive_mu_update(worker &ipm_worker) {
    // compute the normalized complementarity
    // eta = after / before
    scalar_t eta = ipm_worker.post_aff_comp / ipm_worker.prev_aff_comp;
    sig = std::max(0., std::min(1., eta)); // clip
    sig = sig * sig * sig;                 // cubic
    mu = sig * ipm_worker.prev_aff_comp / ipm_worker.n_ipm_cstr;
    assert(mu > 0);
    ipm_reject_corrector = ipm_conditional_corrector && ipm_worker.post_aff_comp > 2 * ipm_worker.prev_aff_comp;
}
} // namespace impl
} // namespace moto