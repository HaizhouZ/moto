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

ns_sqp::solver_graph_type &ns_sqp::restoration_graph() {
    if (!active_model_graph_) {
        throw std::runtime_error("ns_sqp has no active model graph; call create_graph() first");
    }
    if (!restoration_graph_) {
        restoration_graph_ = std::make_shared<solver_graph_type>(graph_n_jobs_);
    }
    model::graph_model(active_model_graph_).realize_into(
        *restoration_graph_,
        [this](const ocp_ptr_t &formulation) {
            solver::restoration::restoration_overlay_settings cfg{
                .rho_u = settings.restoration.rho_u,
                .rho_y = settings.restoration.rho_y,
                .rho_eq = settings.restoration.rho_eq,
                .rho_ineq = settings.restoration.rho_ineq,
                .lambda_reg = settings.restoration.lambda_reg,
            };
            return node_type(solver::restoration::build_restoration_overlay_problem(formulation, cfg), mem_);
        });
    return *restoration_graph_;
}
} // namespace moto
