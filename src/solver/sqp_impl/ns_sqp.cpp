#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <Eigen/Core>
namespace moto {
namespace {
bool same_restoration_cfg(const solver::restoration::restoration_overlay_settings &lhs,
                          const solver::restoration::restoration_overlay_settings &rhs) {
    return lhs.rho_u == rhs.rho_u &&
           lhs.rho_y == rhs.rho_y &&
           lhs.rho_eq == rhs.rho_eq &&
           lhs.rho_ineq == rhs.rho_ineq;
}
} // namespace

ns_sqp::ns_sqp(size_t n_jobs)
    : graph_n_jobs_(std::min(n_jobs, size_t(MAX_THREADS))),
      riccati_solver_(new solver_type()) {
    Eigen::setNbThreads(1);
}

ns_sqp::solver_graph_type &ns_sqp::restoration_graph() {
    if (!active_model_graph_) {
        throw std::runtime_error("ns_sqp has no active model graph; call create_graph() first");
    }
    solver::restoration::restoration_overlay_settings cfg{
        .rho_u = settings.restoration.rho_u,
        .rho_y = settings.restoration.rho_y,
        .rho_eq = settings.restoration.rho_eq,
        .rho_ineq = settings.restoration.rho_ineq,
    };
    const bool needs_rebuild =
        !active_model_graph_->restoration_runtime ||
        active_model_graph_->dirty ||
        !active_model_graph_->restoration_cfg_valid ||
        !same_restoration_cfg(active_model_graph_->restoration_cfg, cfg);
    if (needs_rebuild) {
        active_model_graph_->restoration_runtime = std::make_shared<solver_graph_type>(graph_n_jobs_);
        model::graph_model(active_model_graph_).realize_into(
            *active_model_graph_->restoration_runtime,
            [this, &cfg](const ocp_ptr_t &formulation) {
                return node_type(solver::restoration::build_restoration_overlay_problem(formulation, cfg));
            });
        active_model_graph_->restoration_cfg = cfg;
        active_model_graph_->restoration_cfg_valid = true;
    }
    return *active_model_graph_->restoration_runtime;
}
} // namespace moto
