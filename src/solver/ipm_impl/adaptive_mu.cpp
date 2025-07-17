#include <moto/solver/ipm/ipm_config.hpp>

namespace moto {
namespace ipm_impl {
void ipm_config::adaptive_mu_update(worker &ipm_worker) {
    // compute the normalized complementarity
    // eta = after / before
    scalar_t eta = ipm_worker.after_normalized_comp / ipm_worker.prev_normalized_comp;
    sig = std::max(0., std::min(1., eta)); // clip
    sig = sig * sig * sig;                 // cubic
    mu = sig * ipm_worker.prev_normalized_comp / ipm_worker.n_ipm_cstr;
}
} // namespace ipm_impl
} // namespace moto