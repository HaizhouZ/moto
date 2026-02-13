#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
namespace moto {
ns_sqp::ns_sqp(size_t n_jobs)
    : mem_(impl::data_mgr::create<ns_sqp::data>()), graph_(n_jobs), riccati_solver_(new solver_type()) {
}
ns_sqp::settings_t::settings_t() {
    mu_method = &ns_sqp::settings_t::ipm.mu_method;
}
} // namespace moto