#include <Eigen/Core>
#include <moto/ocp/dynamics/dense_dynamics.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_sqp.hpp>
#include <unordered_map>
#include <utility>
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
void ns_sqp::realize_runtime(storage_type &runtime,
                             const graph_model::interval_snapshot &snapshot,
                             StageBuilder &&stage_builder) {
    runtime.clear();
    const bool add_virtual_initial_state = settings.initial_state == initial_state_mode::optimized;
    if (add_virtual_initial_state && snapshot.intervals->empty()) {
        throw std::runtime_error("initial state optimization requires at least one solver stage");
    }
    runtime.reserve(snapshot.intervals->size() + (add_virtual_initial_state ? 1 : 0));
    if (add_virtual_initial_state) {
        auto virtual_stage = stage_builder(build_initial_state_virtual_stage(snapshot.intervals->front()));
        if (!virtual_stage) {
            throw std::runtime_error("ns_sqp::realize_runtime stage_builder returned null virtual initial stage");
        }
        runtime.add(node_type(virtual_stage, true));
    }
    for (const auto &stage_ocp : *snapshot.intervals) {
        auto built = stage_builder(stage_ocp);
        if (!built) {
            throw std::runtime_error("ns_sqp::realize_runtime stage_builder returned null stage_ocp");
        }
        runtime.add(node_type(built));
    }
    runtime.flatten_nodes();
}

ocp_ptr_t ns_sqp::build_initial_state_virtual_stage(const ocp_ptr_t &first_stage) const {
    if (!first_stage) {
        throw std::runtime_error("ns_sqp::build_initial_state_virtual_stage requires a first stage");
    }
    first_stage->wait_until_ready();
    if (first_stage->num(__x) == 0) {
        throw std::runtime_error("initial state optimization requires the first stage to have x state variables");
    }

    auto virtual_stage = ocp::create();
    virtual_stage->set_automatic_reorder_primal(true);

    var_list dyn_args;
    std::vector<cs::SX> residuals;
    dyn_args.reserve(first_stage->num(__x) * 3);
    residuals.reserve(first_stage->num(__x));

    for (const shared_expr &expr : first_stage->exprs(__x)) {
        const auto x = expr.cast<sym>();
        const var y = x->next();
        const auto u = sym::inputs(fmt::format("moto_initial_state_step_{}", x->name()), x->tdim());
        dyn_args.emplace_back(x);
        dyn_args.emplace_back(y);
        dyn_args.emplace_back(u);
        residuals.emplace_back(x->symbolic_difference(y, x->symbolic_integrate(*x, u)));
    }

    auto dyn = dynamics(new dense_dynamics(
        fmt::format("moto_initial_state_dynamics_{}", first_stage->uid()),
        dyn_args,
        cs::SX::vertcat(residuals),
        approx_order::second,
        __dyn));
    virtual_stage->add(*dyn);
    virtual_stage->wait_until_ready();
    return virtual_stage;
}

void ns_sqp::sync_initial_state_virtual_stage(storage_type &runtime) const {
    auto &nodes = runtime.flatten_nodes();
    if (nodes.size() < 2 || !nodes.front()->internal_initial_state) {
        return;
    }
    data *virtual_initial = nodes.front();
    data *first_real = nodes[1];
    if (virtual_initial->sym_val().value_[__x].size() != first_real->sym_val().value_[__x].size() ||
        virtual_initial->sym_val().value_[__y].size() != first_real->sym_val().value_[__x].size()) {
        throw std::runtime_error("virtual initial stage state layout does not match first real stage x layout");
    }
    virtual_initial->sym_val().value_[__x] = first_real->sym_val().value_[__x];
    virtual_initial->sym_val().value_[__y] = first_real->sym_val().value_[__x];
    virtual_initial->sym_val().value_[__u].setZero();
}

template <typename StageBuilder>
size_t ns_sqp::rebuild_runtime_from_model(storage_type &runtime,
                                          StageBuilder &&stage_builder) {
    const auto snapshot = model_graph_.composed_intervals();
    realize_runtime(runtime, snapshot, std::forward<StageBuilder>(stage_builder));
    return snapshot.revision;
}

ns_sqp::storage_type &ns_sqp::active_data() {
    if (phase_graph_override_ != nullptr) {
        return *phase_graph_override_;
    }

    for (;;) {
        const size_t model_revision = model_graph_.revision();
        const auto initial_state = settings.initial_state;
        if (solver_runtime_revision_.load(std::memory_order_acquire) == model_revision &&
            solver_runtime_initial_state_mode_ == initial_state) {
            return solver_runtime_;
        }

        std::lock_guard<std::mutex> lock(solver_runtime_mutex_);
        if (solver_runtime_revision_.load(std::memory_order_relaxed) == model_graph_.revision() &&
            solver_runtime_initial_state_mode_ == settings.initial_state) {
            return solver_runtime_;
        }

        const size_t built_revision = rebuild_runtime_from_model(
            solver_runtime_,
            [](const ocp_ptr_t &stage_ocp) { return stage_ocp; });
        if (built_revision == model_graph_.revision()) {
            solver_runtime_initial_state_mode_ = settings.initial_state;
            solver_runtime_revision_.store(built_revision, std::memory_order_release);
            return solver_runtime_;
        }
    }
}

std::vector<ns_sqp::data *> &ns_sqp::solver_nodes() {
    auto &nodes = active_data().flatten_nodes();
    thread_local std::unordered_map<const ns_sqp *, std::vector<data *>> public_nodes_by_solver;
    auto &public_solver_nodes_ = public_nodes_by_solver[this];
    public_solver_nodes_.clear();
    public_solver_nodes_.reserve(nodes.size());
    for (data *node : nodes) {
        if (!node->internal_initial_state) {
            public_solver_nodes_.push_back(node);
        }
    }
    return public_solver_nodes_;
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
    const size_t model_revision = model_graph_.revision();
    const bool needs_rebuild =
        restoration_runtime_revision_ != model_revision ||
        restoration_runtime_initial_state_mode_ != settings.initial_state ||
        !restoration_cfg_valid_ ||
        !same_restoration_cfg(restoration_cfg_, cfg);
    if (needs_rebuild) {
        restoration_runtime_revision_ = rebuild_runtime_from_model(
            restoration_runtime_,
            [&cfg](const ocp_ptr_t &stage_ocp) {
                return solver::restoration::build_restoration_overlay_problem(stage_ocp, cfg);
            });
        restoration_cfg_ = cfg;
        restoration_runtime_initial_state_mode_ = settings.initial_state;
        restoration_cfg_valid_ = true;
    }
    return restoration_runtime_;
}

ns_sqp::storage_type &ns_sqp::equality_init_graph() {
    solver::equality_init::equality_init_overlay_settings cfg{
        .rho_eq = settings.eq_init.rho_eq,
    };
    const size_t model_revision = model_graph_.revision();
    const bool needs_rebuild =
        equality_init_runtime_revision_ != model_revision ||
        equality_init_runtime_initial_state_mode_ != settings.initial_state ||
        !equality_init_cfg_valid_ ||
        !same_equality_init_cfg(equality_init_cfg_, cfg);
    if (needs_rebuild) {
        equality_init_runtime_revision_ = rebuild_runtime_from_model(
            equality_init_runtime_,
            [&cfg](const ocp_ptr_t &stage_ocp) {
                return solver::equality_init::build_equality_init_overlay_problem(stage_ocp, cfg);
            });
        equality_init_cfg_ = cfg;
        equality_init_runtime_initial_state_mode_ = settings.initial_state;
        equality_init_cfg_valid_ = true;
    }
    return equality_init_runtime_;
}
} // namespace moto
