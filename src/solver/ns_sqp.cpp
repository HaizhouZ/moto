#include <moto/solver/ns_sqp.hpp>

namespace moto {
ns_sqp::ns_sqp() {
    settings.mu_method = ipm_impl::quality_function_based; // default method
}
} // namespace moto