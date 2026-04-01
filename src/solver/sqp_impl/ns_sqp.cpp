#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <Eigen/Core>
namespace moto {
ns_sqp::ns_sqp(size_t n_jobs)
    : mem_(impl::data_mgr::create<ns_sqp::data>()),
      graph_n_jobs_(std::min(n_jobs, size_t(MAX_THREADS))),
      riccati_solver_(new solver_type()) {
    Eigen::setNbThreads(1);
}
} // namespace moto
