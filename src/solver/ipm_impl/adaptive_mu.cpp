#include <moto/solver/ipm/ipm_config.hpp>

namespace moto {
namespace solver {
void ipm_config::adaptive_mu_update(worker &ipm_worker) {
    // compute the normalized complementarity
    // eta = after / before
    if (ipm_worker.n_ipm_cstr > 0) {
        scalar_t eta = ipm_worker.post_aff_comp / ipm_worker.prev_aff_comp;
        sig = std::max((scalar_t)0., std::min((scalar_t)1., eta)); // clip
        sig = std::pow(sig, 3);                                     // cubic
        mu = sig * ipm_worker.prev_aff_comp / ipm_worker.n_ipm_cstr;
        // fmt::print("comp res before: {:.3e}, after: {:.3e}, eta: {:.3e}, sig: {:.3e}, mu: {:.3e}\n",
        //            ipm_worker.prev_aff_comp / ipm_worker.n_ipm_cstr,
        //            ipm_worker.post_aff_comp / ipm_worker.n_ipm_cstr, eta, sig, mu);
        mu = std::max(mu, (scalar_t)1e-11);
        assert(mu > 0);
        ipm_reject_corrector = ipm_conditional_corrector && ipm_worker.post_aff_comp > 1 * ipm_worker.prev_aff_comp;
    }
}
} // namespace solver
} // namespace moto