#include <moto/solver/ns_sqp.hpp>

namespace moto {
ns_sqp::ns_sqp() {
    settings.mu_method = solver::ipm_config::quality_function_based; // default method
    settings.adaptive_mu_allowed = true;                   // enable adaptive mu
}
} // namespace moto