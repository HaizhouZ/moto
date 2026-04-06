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

class solver_graph_storage_interface final : public graph_model::storage_interface {
  public:
    using graph_type = ns_sqp::storage_type;
    using stage_node_builder_t = std::function<ns_sqp::node_type(const ocp_ptr_t &)>;

    explicit solver_graph_storage_interface(graph_type &graph, stage_node_builder_t stage_node_builder)
        : graph_(graph), stage_node_builder_(std::move(stage_node_builder)) {}

    void clear() override {
        graph_.clear();
        node_refs_.clear();
    }

    size_t add_stage(const ocp_ptr_t &stage_ocp) override {
        node_refs_.emplace_back(&graph_.add(stage_node_builder_(stage_ocp)));
        return node_refs_.size() - 1;
    }

    void connect(size_t st_id, size_t ed_id) override {
        graph_.connect(*node_refs_.at(st_id), *node_refs_.at(ed_id), {2, true, true});
    }

    void set_head(size_t node_id) override {
        graph_.set_head(*node_refs_.at(node_id));
    }

    void set_tail(size_t node_id) override {
        graph_.set_tail(*node_refs_.at(node_id));
    }

  private:
    graph_type &graph_;
    stage_node_builder_t stage_node_builder_;
    std::vector<ns_sqp::node_type *> node_refs_;
};
} // namespace

ns_sqp::ns_sqp(size_t n_jobs)
    : graph_n_jobs_(std::min(n_jobs, size_t(MAX_THREADS))),
      riccati_solver_(new solver_type()) {
    Eigen::setNbThreads(1);
}

ns_sqp::storage_type &ns_sqp::active_data() {
    if (phase_graph_override_ != nullptr) {
        return *phase_graph_override_;
    }
    if (!active_model_graph_) {
        throw std::runtime_error("ns_sqp has no active model graph; call create_graph() first");
    }
    if (!solver_runtime_) {
        solver_runtime_ = std::make_shared<storage_type>(graph_n_jobs_);
    }
    if (active_model_graph_->topology_changed()) {
        solver_graph_storage_interface realization(*solver_runtime_, [this](const ocp_ptr_t &stage_ocp) {
            return node_type(stage_ocp);
        });
        active_model_graph_->realize_into(realization);
    }
    return *solver_runtime_;
}

ns_sqp::storage_type &ns_sqp::restoration_graph() {
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
        !restoration_runtime_ ||
        active_model_graph_->topology_changed() ||
        !restoration_cfg_valid_ ||
        !same_restoration_cfg(restoration_cfg_, cfg);
    if (needs_rebuild) {
        restoration_runtime_ = std::make_shared<storage_type>(graph_n_jobs_);
        solver_graph_storage_interface realization(*restoration_runtime_,
                                                   [this](const ocp_ptr_t &stage_ocp) {
                                                       return node_type(stage_ocp);
                                                   });
        active_model_graph_->realize_into(
            realization,
            [&cfg](const ocp_ptr_t &stage_ocp) {
                return solver::restoration::build_restoration_overlay_problem(stage_ocp, cfg);
            });
        restoration_cfg_ = cfg;
        restoration_cfg_valid_ = true;
    }
    return *restoration_runtime_;
}

ns_sqp::storage_type &ns_sqp::equality_init_graph() {
    if (!active_model_graph_) {
        throw std::runtime_error("ns_sqp has no active model graph; call create_graph() first");
    }
    solver::equality_init::equality_init_overlay_settings cfg{
        .rho_eq = settings.eq_init.rho_eq,
    };
    const bool needs_rebuild =
        !equality_init_runtime_ ||
        active_model_graph_->topology_changed() ||
        !equality_init_cfg_valid_ ||
        !same_equality_init_cfg(equality_init_cfg_, cfg);
    if (needs_rebuild) {
        equality_init_runtime_ = std::make_shared<storage_type>(graph_n_jobs_);
        solver_graph_storage_interface realization(*equality_init_runtime_,
                                                   [this](const ocp_ptr_t &stage_ocp) {
                                                       return node_type(stage_ocp);
                                                   });
        active_model_graph_->realize_into(
            realization,
            [&cfg](const ocp_ptr_t &stage_ocp) {
                return solver::equality_init::build_equality_init_overlay_problem(stage_ocp, cfg);
            });
        equality_init_cfg_ = cfg;
        equality_init_cfg_valid_ = true;
    }
    return *equality_init_runtime_;
}
} // namespace moto
