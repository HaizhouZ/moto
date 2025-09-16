#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
namespace moto {
ns_sqp::ns_sqp(size_t n_jobs)
    : mem_(impl::data_mgr::create<ns_sqp::data>()), graph_(n_jobs), riccati_solver_(new solver_type()) {
    settings.mu_method = solver::ipm_config::quality_function_based; // default method
    settings.adaptive_mu_allowed = true;                             // enable adaptive mu
}
} // namespace moto