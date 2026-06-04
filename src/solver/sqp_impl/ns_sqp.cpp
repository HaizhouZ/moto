#include <Eigen/Core>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
namespace moto {
namespace {
bool same_restoration_cfg(const solver::restoration::restoration_overlay_settings &lhs,
                          const solver::restoration::restoration_overlay_settings &rhs) {
    return lhs.rho_u == rhs.rho_u &&
           lhs.rho_y == rhs.rho_y &&
           lhs.rho_eq == rhs.rho_eq &&
           lhs.rho_ineq == rhs.rho_ineq;
}

bool same_equality_init_cfg(const solver::equality_init::equality_init_overlay_settings &lhs,
                            const solver::equality_init::equality_init_overlay_settings &rhs) {
    return lhs.rho_eq == rhs.rho_eq;
}
} // namespace

ns_sqp::ns_sqp(size_t n_jobs)
    : graph_n_jobs_(normalize_parallel_jobs(n_jobs)),
      solver_runtime_(graph_n_jobs_),
      restoration_runtime_(graph_n_jobs_),
      equality_init_runtime_(graph_n_jobs_) {
    Eigen::setNbThreads(1);
}

template <typename StageBuilder>
void ns_sqp::realize_runtime(storage_type &runtime, StageBuilder &&stage_builder) {
    runtime.clear();
    const size_t num_edges = model_graph_.edges_.size();
    if (num_edges == 0) {
        throw std::runtime_error("graph_model expects a non-empty path");
    }
    runtime.reserve(num_edges);
    for (size_t eid = 0; eid < num_edges; ++eid) {
        const auto &edge = model_graph_.edges_[eid];
        if (!edge) {
            throw std::runtime_error("graph_model found null edge");
        }
        const auto stage_ocp = std::static_pointer_cast<ocp>(
            model_graph_.compose_interval(edge, eid + 1 == num_edges));
        auto built = stage_builder(stage_ocp);
        if (!built) {
            throw std::runtime_error("ns_sqp::realize_runtime stage_builder returned null stage_ocp");
        }
        runtime.add(node_type(built));
    }
}

ns_sqp::storage_type &ns_sqp::active_data() {
    if (phase_graph_override_ != nullptr) {
        return *phase_graph_override_;
    }
    const size_t model_revision = model_graph_.revision_;
    if (solver_runtime_revision_ != model_revision) {
        realize_runtime(solver_runtime_, [](const ocp_ptr_t &stage_ocp) { return stage_ocp; });
        solver_runtime_revision_ = model_revision;
    }
    return solver_runtime_;
}

std::vector<ns_sqp::data *> &ns_sqp::solver_nodes() {
    return active_data().flatten_nodes();
}

ns_sqp::scoped_phase_graph_override::scoped_phase_graph_override(ns_sqp &owner,
                                                                 storage_type &graph,
                                                                 bool in_restoration)
    : owner(owner), in_restoration_backup(owner.settings.in_restoration) {
    use_graph(graph, in_restoration);
}

void ns_sqp::scoped_phase_graph_override::use_graph(storage_type &graph,
                                                    bool in_restoration) {
    owner.settings.in_restoration = in_restoration;
    owner.phase_graph_override_ = &graph;
}

void ns_sqp::scoped_phase_graph_override::use_default_graph(bool in_restoration) {
    owner.settings.in_restoration = in_restoration;
    owner.phase_graph_override_ = nullptr;
}

ns_sqp::scoped_phase_graph_override::~scoped_phase_graph_override() {
    owner.settings.in_restoration = in_restoration_backup;
    owner.phase_graph_override_ = nullptr;
}

ns_sqp::storage_type &ns_sqp::restoration_graph() {
    solver::restoration::restoration_overlay_settings cfg{
        .rho_u = settings.restoration.rho_u,
        .rho_y = settings.restoration.rho_y,
        .rho_eq = settings.restoration.rho_eq,
        .rho_ineq = settings.restoration.rho_ineq,
    };
    const size_t model_revision = model_graph_.revision_;
    const bool needs_rebuild =
        restoration_runtime_revision_ != model_revision ||
        !restoration_cfg_valid_ ||
        !same_restoration_cfg(restoration_cfg_, cfg);
    if (needs_rebuild) {
        realize_runtime(restoration_runtime_, [&cfg](const ocp_ptr_t &stage_ocp) {
            return solver::restoration::build_restoration_overlay_problem(stage_ocp, cfg);
        });
        restoration_runtime_revision_ = model_revision;
        restoration_cfg_ = cfg;
        restoration_cfg_valid_ = true;
    }
    return restoration_runtime_;
}

ns_sqp::storage_type &ns_sqp::equality_init_graph() {
    solver::equality_init::equality_init_overlay_settings cfg{
        .rho_eq = settings.eq_init.rho_eq,
    };
    const size_t model_revision = model_graph_.revision_;
    const bool needs_rebuild =
        equality_init_runtime_revision_ != model_revision ||
        !equality_init_cfg_valid_ ||
        !same_equality_init_cfg(equality_init_cfg_, cfg);
    if (needs_rebuild) {
        realize_runtime(equality_init_runtime_, [&cfg](const ocp_ptr_t &stage_ocp) {
            return solver::equality_init::build_equality_init_overlay_problem(stage_ocp, cfg);
        });
        equality_init_runtime_revision_ = model_revision;
        equality_init_cfg_ = cfg;
        equality_init_cfg_valid_ = true;
    }
    return equality_init_runtime_;
}
} // namespace moto
