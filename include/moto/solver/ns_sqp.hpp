#ifndef __NS_SQP__
#define __NS_SQP__

#include <moto/model/graph_model.hpp>
#include <moto/ocp/constr.hpp>
#include <moto/ocp/impl/shooting_node.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>
#include <functional>

namespace moto {
namespace solver {
namespace ns_riccati {
struct generic_solver;
}
} // namespace solver
struct ns_sqp {

    struct scaling_settings {
        enum class mode_t : size_t {
            none,       ///< no scaling
            gradient,   ///< row-normalise each Jacobian to unit inf-norm
            equilibrium ///< Ruiz / Sinkhorn doubly-balanced scaling
        } mode = mode_t::gradient;
        size_t equilibrium_iters = 5;          ///< Ruiz iterations (equilibrium only)
        scalar_t min_scale = 1e-6;             ///< clamp to avoid division by zero
        scalar_t update_ratio_threshold = 10.; ///< re-scale when dual_res / prim_res >= this; below it cached scales are reused
    };

    struct iterative_refinement_setting {
        bool enabled = true;           ///< whether to use iterative refinement
        size_t max_iters = 5;          ///< max refinement iterations
        scalar_t prim_res_tol = 1e-10; ///< primal residual tolerance for refinement
        scalar_t dual_res_tol = 1e-10; ///< dual residual tolerance for refinement
    };

    struct linesearch_setting : public solver::linesearch_config {
        bool enabled = true;       ///< whether to use line search
        size_t max_steps = 5;      ///< max line search steps
        bool enable_soc = true;    ///< whether to try a second-order correction before backtracking
        size_t max_soc_iter = 4;   ///< max number of second-order correction retries per SQP iteration (p_max in IPOPT)
        scalar_t kappa_soc = 0.99; ///< SOC abort threshold: abort if θ(x_soc) > kappa_soc * θ_soc_old (IPOPT eq. A-5.9)
        enum class failure_backup_strategy : size_t {
            min_step,   ///< reset to the minimum step size
            best_trial, ///< reset to the best trial so far
        } failure_strategy = failure_backup_strategy::best_trial;

        enum class backtrack_scheme_t : size_t {
            linspace,  ///< alpha decreases by alpha_init / max_steps each step (uniform spacing)
            geometric, ///< alpha *= backtrack_factor each step (exponential decay)
        } backtrack_scheme = backtrack_scheme_t::linspace;
        scalar_t backtrack_factor = 0.5; ///< geometric backtracking reduction factor (used when backtrack_scheme == geometric)

        enum class search_method : size_t {
            filter,             ///< IPOPT-style 2-objective filter line search (default)
            merit_backtracking, ///< merit function: ||constraint violation||^2 + sigma * ||dual residual||^2
        } method = search_method::filter;

        scalar_t primal_gamma = 1e-4;        ///< 2-obj filter primal improvement requirement: higher-> stricter
        scalar_t dual_gamma = 1e-4;          ///< IPOPT-like filter objective improvement requirement: higher-> stricter
        scalar_t constr_vio_min_frac = 1e-4; ///< Threshold for switching condition (fraction of initial primal residual), lower than this * initial constraint violation means we are close enough to the feasible region to switch to objective decrease mode in line search
        scalar_t armijo_dec_frac = 1e-4;     ///< Sufficient decrease tolerance (eta in Armijo condition), smaller -> more strict decrease requirement
        scalar_t s_phi = 2.3;                ///< IPOPT switching condition exponent on objective decrease (s_phi in IPOPT paper)
        scalar_t s_theta = 1.1;              ///< IPOPT switching condition exponent on constraint violation (s_theta in IPOPT paper)
        scalar_t merit_sigma = 1.0;          ///< merit_backtracking weight on ||dual residual||^2 relative to ||constraint violation||^2

        // Flat-objective accept: accept when the directional derivative is negligibly small,
        // the primal residual is already low, and the step is non-trivial (so we make dual progress).
        bool enable_flat_obj_accept = true;
        scalar_t flat_obj_dec_tol = 1e-2;   ///< |fullstep_dec| below this is considered "flat"
        scalar_t flat_obj_prim_tol = 1e-6;  ///< primal residual must be below this to trigger
        scalar_t flat_obj_step_tol = 1e-12; ///< step norm must exceed this to be "non-trivial"
    };

    struct restoration_settings {
        bool enabled = true;  ///< whether restoration mode is allowed
        size_t max_iter = 50; ///< max restoration iterations per trigger
        /// trigger restoration when the filter line search fails this many consecutive outer SQP iters
        size_t trigger_on_failure_count = 3;
        scalar_t rho_u = 1e-4; ///< proximal weight on u (anchors to point where restoration was triggered)
        scalar_t rho_y = 1e-4; ///< proximal weight on y (anchors to point where restoration was triggered)
        /// dual regularization for GN equality constraints: Hess += (1/rho_eq)*J^T*J;
        /// dlam = (J*du + h) / rho_eq. Smaller -> tighter constraint satisfaction per step.
        scalar_t rho_eq = 1e-6;
        /// exit restoration when inf_prim_res drops below this fraction of the entry infeasibility
        scalar_t restoration_improvement_frac = 0.9;
    };

    struct ipm_config : public solver::ipm_config {
        scalar_t mu0 = 1.0;        ///< initial barrier parameter
        bool warm_start = false;   ///< whether to warm start the IPM solver
        bool globalization = true; ///< whether to use IPM globalization

        scalar_t mu_monotone_fraction_threshold = 10.0; ///< threshold for monotone decrease of mu, smaller -> more likely to use monotone decrease
        scalar_t mu_monotone_factor = 0.2;              ///< factor for monotonic decrease of mu, smaller -> faster decrease
    };

    struct settings_t : public workspace_data_collection<linesearch_setting, ipm_config> {
        using base = workspace_data_collection<linesearch_setting, ipm_config>;
        using worker = typename base::worker;

        linesearch_setting &ls;
        ipm_config &ipm;
        size_t max_iter = 100;  ///< maximum number of SQP iterations
        double prim_tol = 1e-6; ///< primal feasibility tolerance
        double dual_tol = 1e-4; ///< dual feasibility tolerance
        double comp_tol = 1e-6; ///< complementarity feasibility tolerance
        double s_max = 100.;    ///< IPOPT-style dual scaling: s_d = max(s_max, ||λ||_1 / n_constr) / s_max

        iterative_refinement_setting rf;
        scaling_settings scaling;
        restoration_settings restoration;

        bool no_except = false;

        settings_t()
            : ls(static_cast<linesearch_setting &>(*this)), ipm(static_cast<ipm_config &>(*this)) {}

      private:
        friend class ns_sqp;
        bool verbose = true;
        size_t n_worker = MAX_THREADS; ///< number of worker threads
        bool in_restoration = false;   ///< whether currently in restoration mode (used to adjust printouts and possibly other settings)
        bool has_ineq_soft = false;    ///< whether the problem has inequality constraints (used to adjust printouts and possibly other settings)
        bool initialized = false;     ///< whether the settings have been initialized based on the problem (used to trigger one-time initialization in the first iteration, e.g. setting initial mu based on initial residuals)
    } settings;

    using solver_type = solver::ns_riccati::generic_solver;
    using ns_riccati_data = solver::ns_riccati::ns_riccati_data;

    struct data : public node_data, ns_riccati_data {
        data(const ocp_ptr_t &prob)
            : node_data(prob), ns_riccati_data((node_data *)this) {
            for (auto f : primal_fields) {
                auto &diag_panels = this->dense().lag_hess_[f][f].diag_panels_;
                if (!diag_panels.empty()) {
                    primal_prox_hess_diagonal_[f].reset(diag_panels.back().data_);
                }
            }
        }
        data(data &&rhs) = default;
        static void update_approx(data *d) {
            d->update_approximation();
        }
        array_type<aligned_vector_map_t, primal_fields> primal_prox_hess_diagonal_;
        /// row scale applied to each constraint field (empty ⟹ scaling not yet applied)
        array_type<vector, constr_fields> scale_c_;
        /// scale applied to each primal field's cost gradient
        array<scalar_t, field::num_prim> scale_p_{};
        /// whether scaling has been applied (and therefore duals must be unscaled)
        bool scaling_applied_ = false;
    };

    enum iter_result_t : size_t {
        unknown = 0,
        success,               ///< converged to a KKT point within tolerances
        exceed_max_iter,       ///< reached maximum number of iterations without convergence
        restoration_failed,    ///< restoration was triggered but failed to make sufficient progress
        infeasible_stationary, ///< reached an infeasible stationary point (e.g. due to LICQ failure) and cannot make progress
    };

    struct kkt_info {
        iter_result_t result = iter_result_t::unknown;
        size_t num_iter = 0; // number of iterations
        size_t ls_steps = 0; ///< line search steps

        scalar_t cost = 0.;                // pure running cost (sum of __cost terms)
        scalar_t log_slack_sum = 0.;       // sum(log(slack)) across all IPM constraints, mu-free
        scalar_t barrier_dir_deriv = 0.;   // sum(d_slack / slack_current) across all IPM constraints, mu-free
        scalar_t objective = 0.;           // barrier objective: cost - mu * log_slack_sum (computed with current mu)
        scalar_t obj_fullstep_dec = 0.;    // cost gradient dot full step (mu-free; combine with barrier_dir_deriv at current mu for full barrier directional derivative)
        scalar_t inf_prim_res = 0.;        // primal residual (constraint violation), inf-norm across all nodes/constraints
        scalar_t prim_res_l1 = 0.;         // primal residual L1 norm (sum of |v| across all nodes/constraints)
        scalar_t inf_dual_res = 0.;        // dual residual (stationary condition)
        scalar_t inf_comp_res = 0.;        // (inequality) complementarity residual
        scalar_t max_diag_scaling = 0.;    // max Nesterov-Todd scaling (lambda/slack) across all IPM constraints
        scalar_t max_eq_dual_norm = 0.;    // max inf-norm of equality (hard) dual variables across all nodes
        scalar_t max_ineq_dual_norm = 0.;  // max inf-norm of inequality dual variables across all nodes
        scalar_t max_dual_norm = 0.;       // max inf-norm of dual variables across all nodes and constraint fields
        scalar_t inf_prim_step = 0.;       // infinity norm of the primal step
        scalar_t inf_dual_step = 0.;       // infinity norm of the dual step (all constraints)
        scalar_t inf_eq_dual_step = 0.;    // infinity norm of the dual step (equality constraints only)
        scalar_t inf_ineq_dual_step = 0.;  // infinity norm of the dual step (inequality constraints only)
        scalar_t inf_dyn_dual_step = 0.;   // inf-norm of dual step for __dyn
        scalar_t inf_eq_x_dual_step = 0.;  // inf-norm of dual step for __eq_x
        scalar_t inf_eq_xu_dual_step = 0.; // inf-norm of dual step for __eq_xu
        scalar_t avg_dual_res = 0.;        // average dual residual: L1 norm of stationarity gradient / number of elements (unscaled)
    } kkt_last;
    kkt_info update(size_t n_iter, bool verbose = true);

    ns_sqp(size_t n_jobs = MAX_THREADS);
    ns_sqp(const ns_sqp &) = delete;
    ~ns_sqp() = default;
    using node_type = impl::shooting_node<data>;

    struct model_graph : public model::graph_model {
        model_graph(ns_sqp &owner, std::shared_ptr<model::graph_model_state> state)
            : model::graph_model(std::move(state)), owner_(&owner) {}

        std::vector<data *> &flatten_nodes() {
            owner_->ensure_realized(state_ptr());
            return owner_->graph_.flatten_nodes();
        }

        ns_sqp *owner_;
    };

    void reset_riccati_solver(solver_type *s) {
        riccati_solver_.reset(s);
    }

    model_graph create_graph() {
        active_model_graph_ = std::make_shared<model::graph_model_state>();
        return model_graph(*this, active_model_graph_);
    }

    static bool has_terminal_terms(const node_ocp_ptr_t &node_prob) {
        if (!node_prob) {
            return false;
        }
        for (size_t f = 0; f < field::num; ++f) {
            for (const shared_expr &expr : node_prob->exprs(f)) {
                if (const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get())) {
                    if (cost_expr->terminal_add()) {
                        return true;
                    }
                }
                if (const auto *constr_expr = dynamic_cast<const generic_constr *>(expr.get())) {
                    if (constr_expr->terminal_add()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static bool is_terminal_term(const shared_expr &expr) {
        if (const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get())) {
            return cost_expr->terminal_add();
        }
        if (const auto *constr_expr = dynamic_cast<const generic_constr *>(expr.get())) {
            return constr_expr->terminal_add();
        }
        return false;
    }

    node_type create_node(const ocp_ptr_t &formulation) {
        return create_node(formulation, ocp::active_status_config{});
    }

    std::vector<node_type> create_nodes(const ocp_ptr_t &formulation,
                                        const std::vector<ocp::active_status_config> &configs) {
        std::vector<node_type> nodes;
        nodes.reserve(configs.size());
        for (const auto &config : configs) {
            nodes.emplace_back(create_node(formulation, config));
        }
        return nodes;
    }

    node_type create_node(const ocp_ptr_t &formulation, const ocp::active_status_config &config) {
        auto cloned = std::dynamic_pointer_cast<ocp>(formulation->clone_base(config));
        if (!cloned) {
            throw std::runtime_error("ns_sqp::create_node failed to clone formulation as ocp");
        }
        cloned->wait_until_ready();
        return node_type(cloned, mem_);
    }

    node_type create_node(const model::model_edge_ptr_t &edge_model) {
        return create_node(edge_model, ocp::active_status_config{});
    }

    std::vector<node_type> create_nodes(const model::model_edge_ptr_t &edge_model,
                                        const std::vector<ocp::active_status_config> &configs) {
        std::vector<node_type> nodes;
        nodes.reserve(configs.size());
        for (const auto &config : configs) {
            nodes.emplace_back(create_node(edge_model, config));
        }
        return nodes;
    }

    node_type create_node(const model::model_edge_ptr_t &edge_model, const ocp::active_status_config &config) {
        auto composed = compose_regular_edge(edge_model, config, false);
        return node_type(std::static_pointer_cast<ocp>(composed), mem_);
    }

    node_type create_terminal_node(const model::model_node_ptr_t &node_model) {
        if (!node_model) {
            throw std::runtime_error("ns_sqp::create_terminal_node received a null model_node");
        }
        node_model->wait_until_ready();
        auto composed = node_model->compose_terminal();
        sanitize_terminal_node(*composed);
        composed->wait_until_ready();
        return node_type(std::static_pointer_cast<ocp>(composed), mem_);
    }

    node_type create_terminal_node(const model::model_edge_ptr_t &edge_model) {
        if (!edge_model) {
            throw std::runtime_error("ns_sqp::create_terminal_node received a null model_edge");
        }
        auto composed = compose_regular_edge(edge_model, {}, true, true);
        return node_type(std::static_pointer_cast<ocp>(composed), mem_);
    }

    std::vector<data *> &flatten_nodes() {
        ensure_realized();
        return graph_.flatten_nodes();
    }

    impl::data_mgr mem_;
    directed_graph<node_type> graph_;

  private:
    static bool terminal_term_depends_on_u(const shared_expr &expr) {
        if (!is_terminal_term(expr)) {
            return false;
        }
        const auto *func = dynamic_cast<const generic_func *>(expr.get());
        if (func == nullptr) {
            return false;
        }
        return std::any_of(func->in_args().begin(), func->in_args().end(), [](const sym &arg) {
            return arg.field() == __u;
        });
    }

    static void sanitize_terminal_node(node_ocp &prob) {
        ocp::active_status_config config;
        for (size_t f = 0; f < field::num; ++f) {
            for (const shared_expr &expr : prob.exprs(f)) {
                if (!terminal_term_depends_on_u(expr)) {
                    continue;
                }
                fmt::print(stderr,
                           "warning: terminal node term {} depends on u and cannot be applied on a terminal x/u node; ignoring it\n",
                           expr->name());
                config.deactivate_list.emplace_back(*expr);
            }
        }
        if (!config.empty()) {
            prob.update_active_status(config, false);
        }
    }

    static edge_ocp_ptr_t compose_regular_edge(const model::model_edge_ptr_t &edge_model,
                                               const ocp::active_status_config &config = {},
                                               bool materialize_sink_terms = false,
                                               bool include_terminal_sink_terms = false) {
        if (!edge_model) {
            throw std::runtime_error("ns_sqp::compose_regular_edge received a null model_edge");
        }
        edge_model->wait_until_ready();
        auto st_node_prob = edge_model->st_node_prob();
        if (st_node_prob) {
            st_node_prob->wait_until_ready();
        }
        if (!config.empty()) {
            st_node_prob = st_node_prob ? st_node_prob->clone_node(config) : node_ocp_ptr_t{};
        }
        auto composed = edge_ocp::compose(
            st_node_prob,
            edge_model->clone_edge(),
            node_ocp_ptr_t{},
            false);
        if (materialize_sink_terms) {
            if (const auto &sink_node = edge_model->ed_node_prob()) {
                for (size_t f = 0; f < field::num; ++f) {
                    if (f == __dyn) {
                        continue;
                    }
                    for (const shared_expr &expr : sink_node->exprs(f)) {
                        const bool terminal_term = is_terminal_term(expr);
                        const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get());
                        const auto *constr_expr = dynamic_cast<const generic_constr *>(expr.get());
                        const auto *func = dynamic_cast<const generic_func *>(expr.get());
                        if ((cost_expr == nullptr && constr_expr == nullptr) || func == nullptr) {
                            continue;
                        }
                        if (terminal_term && !include_terminal_sink_terms) {
                            continue;
                        }
                        bool lowerable_term = true;
                        bool needs_lower_to_y = false;
                        bool has_u = false;
                        for (const sym &arg : func->in_args()) {
                            if (arg.field() == __x) {
                                needs_lower_to_y = true;
                                continue;
                            }
                            if (arg.field() == __y || arg.field() == __p) {
                                continue;
                            }
                            if (arg.field() == __u) {
                                has_u = true;
                            }
                            lowerable_term = false;
                            break;
                        }
                        if (terminal_term && has_u) {
                            fmt::print(stderr,
                                       "warning: terminal node term {} depends on u and cannot be lowered onto the final edge; ignoring it\n",
                                       expr->name());
                            continue;
                        }
                        if (!terminal_term && f != __cost) {
                            continue;
                        }
                        if (!lowerable_term) {
                            continue;
                        }
                        auto lowered = expr.clone();
                        auto *lowered_func = dynamic_cast<generic_func *>(lowered.get());
                        if (lowered_func == nullptr) {
                            continue;
                        }
                        if (needs_lower_to_y) {
                            for (const sym &arg : lowered_func->in_args()) {
                                if (arg.field() == __x) {
                                    lowered_func->substitute_argument(arg, arg.next());
                                }
                            }
                        }
                        composed->add(lowered);
                    }
                }
            }
        }
        composed->wait_until_ready();
        return composed;
    }

    void ensure_realized() {
        ensure_realized(active_model_graph_);
    }

    void ensure_realized(const std::shared_ptr<model::graph_model_state> &state) {
        if (!state || !state->dirty) {
            return;
        }
        graph_.clear();
        const size_t num_nodes = state->nodes.size();
        const size_t num_edges = state->edges.size();
        std::vector<node_type *> solver_nodes_by_edge(num_edges, nullptr);
        std::vector<std::vector<size_t>> incoming_edge_ids(num_nodes);
        std::vector<std::vector<size_t>> outgoing_edge_ids(num_nodes);

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            const auto &edge_h = state->edges.at(edge_id);
            if (!edge_h) {
                throw std::runtime_error("ns_sqp::ensure_realized encountered a null model_edge");
            }
            incoming_edge_ids.at(edge_h->ed()->id()).push_back(edge_id);
            outgoing_edge_ids.at(edge_h->st()->id()).push_back(edge_id);
        }

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            const auto &edge_h = state->edges.at(edge_id);
            const bool sink_without_outgoing = outgoing_edge_ids.at(edge_h->ed()->id()).empty();
            const auto formulation = compose_regular_edge(edge_h,
                                                          {},
                                                          sink_without_outgoing,
                                                          sink_without_outgoing);
            auto &solver_node = graph_.add(node_type(std::static_pointer_cast<ocp>(formulation), mem_));
            solver_nodes_by_edge[edge_id] = &solver_node;
        }

        std::vector<node_type *> head_candidates;
        std::vector<node_type *> tail_candidates;
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            const auto &incoming = incoming_edge_ids.at(node_id);
            const auto &outgoing = outgoing_edge_ids.at(node_id);
            if (incoming.empty() && outgoing.empty()) {
                continue;
            }
            if (incoming.empty()) {
                if (outgoing.size() != 1) {
                    throw std::runtime_error("ns_sqp::ensure_realized currently expects a unique outgoing edge from the source model node");
                }
                head_candidates.push_back(solver_nodes_by_edge.at(outgoing.front()));
                continue;
            }
            if (outgoing.empty()) {
                if (incoming.size() != 1) {
                    throw std::runtime_error("ns_sqp::ensure_realized currently expects a unique incoming edge for a sink model node");
                }
                tail_candidates.push_back(solver_nodes_by_edge.at(incoming.front()));
                continue;
            }
            for (const size_t incoming_edge_id : incoming) {
                for (const size_t outgoing_edge_id : outgoing) {
                    graph_.connect(*solver_nodes_by_edge.at(incoming_edge_id),
                                   *solver_nodes_by_edge.at(outgoing_edge_id),
                                   {2, true, true});
                }
            }
        }

        if (head_candidates.size() != 1) {
            throw std::runtime_error("ns_sqp::ensure_realized expects a single source path in model_graph");
        }
        if (tail_candidates.size() != 1) {
            throw std::runtime_error("ns_sqp::ensure_realized expects a single sink path in model_graph");
        }
        graph_.set_head(*head_candidates.front());
        graph_.set_tail(*tail_candidates.front());
        state->dirty = false;
    }

    std::unique_ptr<solver_type> riccati_solver_ = nullptr;
    std::shared_ptr<model::graph_model_state> active_model_graph_;

    template <typename worker_type>
    struct stacked_workers : public std::vector<worker_type> {
        void reset(size_t n) {
            this->clear();
            this->reserve(n);
            for (size_t i = 0; i < n; ++i) {
                this->emplace_back();
            }
        }
    };

    stacked_workers<settings_t::worker> setting_per_thread;

    /// print inf norms of constraint residuals and Jacobians across all nodes, to diagnose scaling
    void print_scaling_info();
    /// print per-field contribution to dual stationarity residual (first node only)
    void print_dual_res_breakdown();
    /// check LICQ at the current point: stacks all constraint Jacobians per node and reports rank via SVD
    void print_licq_info();
    /// print statistics header
    void print_stat_header();
    /// print statistics for the current iteration
    void print_stats(const kkt_info &info);
    /// compute the kkt information of the current solution
    kkt_info compute_kkt_info(bool update_dual_res = true);
    /// perform iterative refinement to improve the solution accuracy, will modify the current solution in place
    void iterative_refinement();
    /// update the line search bounds with the (probably updated) max value
    void finalize_ls_bound_and_set_to_max();
    /**
     * @brief Compute row scales from current Jacobian magnitudes (gradient or equilibrium mode)
     *        and apply them in-place to v_, jac_ and the cost gradient.
     *        Must be called after update_approximation(eval_derivatives).
     * @param kkt current KKT info (used to decide whether to refresh the scale vectors)
     */
    void compute_and_apply_scaling(const kkt_info &kkt);
    /// Reverse the row scaling on the dual variables after the QP solve.
    void unscale_duals();
    /// Clear all stored scales (called on initialize()).
    void reset_scaling();
    /**
     * @brief filter line search for the current iteration, will update the line search data and the kkt info of the current solution
     * @note just for convenient reset
     */
    struct filter_linesearch_per_iter_data {
        bool recompute_approx = true;
        bool stop = false;                                                  ///< whether to stop the line search
        bool enforce_min = false;                                           ///< whether to enforce the minimum step size
        size_t soc_iter_cnt = 0;                                            ///< number of second-order correction attempts in the current SQP iteration
        bool skip_soc = false;                                              ///< whether to skip second-order correction
        scalar_t theta_soc_old = std::numeric_limits<scalar_t>::infinity(); ///< θ(x_k) at SOC entry, for κ_soc abort check (IPOPT A-5.9)
        size_t step_cnt = 0;                                                ///< current line search step
        scalar_t initial_alpha_primal = 0.;
        scalar_t initial_alpha_dual = 0.;
        bool switching_condition = false; ///< whether the switching condition for line search is met (used to decide whether to require Armijo decrease in the line search)
        bool armijo_cond_met = false;     ///< whether the Armijo condition is met for the current trial step
        void reset_per_iter_data() {
            new (this) filter_linesearch_per_iter_data();
        }
    };
    struct filter_linesearch_data : public filter_linesearch_per_iter_data {
        /***** filter part *****/
        struct point {
            scalar_t prim_res = std::numeric_limits<scalar_t>::infinity();
            scalar_t dual_res = std::numeric_limits<scalar_t>::infinity();
            scalar_t objective = std::numeric_limits<scalar_t>::infinity();
            /// @check if a point is in the filter
            bool in_filter(const point &filter_entry, const settings_t &settings) const;
        };
        struct trial : public point, public solver::linesearch_config {
        } best_trial;
        std::vector<point> points;                                           ///< filter for accepting line search steps
        scalar_t constr_vio_min = std::numeric_limits<scalar_t>::infinity(); ///< constraint violation bound for switching condition in line search
        bool last_step_was_armijo = false;
        size_t filter_reject_cnt = 0; ///< number of consecutive filter rejections, used for adaptive strategies in line search

        void update_filter(const kkt_info &kkt, settings_t &settings);
        bool try_step(const kkt_info &trial_kkt, const kkt_info &current_kkt, settings_t &settings);

        /***** merit backtracking part (used when settings.ls.method == merit_backtracking) *****/
        scalar_t merit_fullstep = std::numeric_limits<scalar_t>::infinity(); ///< merit value at full step (alpha=1), for directional derivative estimate
        struct merit_trial {
            scalar_t merit = std::numeric_limits<scalar_t>::infinity();
            scalar_t alpha_primal = 0.;
            scalar_t alpha_dual = 0.;
        } best_merit_trial;
    };

    enum class line_search_action {
        accept,
        backtrack,
        retry_second_order_correction,
        stop,
    };

    struct iteration_context {
        kkt_info current;
        kkt_info trial;
        line_search_action action = line_search_action::accept;
        bool mu_changed = false;
    };

    void step_back_alpha(filter_linesearch_per_iter_data &ls);
    line_search_action filter_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt);
    line_search_action merit_linesearch(filter_linesearch_data &ls, const kkt_info &trial_kkt, const kkt_info &current_kkt);

    void second_order_correction();
    void ineq_constr_correction(iteration_context &ctx);
    void ineq_constr_prediction();
    /// initialize the solver before the first iteration or after a reset, returns the initial kkt info
    kkt_info initialize();
    kkt_info restoration_update(const kkt_info &kkt_before, filter_linesearch_data &ls);
    /// Compute restoration objective J_rest = ½Σ(‖F_0‖² + ‖s_c_stacked_0_k‖²) across all nodes.
    /// Used to track filter progress in the restoration phase instead of the original running cost.
    size_t ls_failure_count_ = 0; ///< consecutive outer iterations where line search produced no improvement
    void post_factorization_correction_step();
    void finalize_correction(data *d);
    void reset_ls_workers();
    void refresh_ls_bounds();
    template <typename Prepare, typename Finalize>
    void run_correction_step(Prepare &&prepare, Finalize &&finalize) {
        graph_.for_each_parallel(
            [prepare = std::forward<Prepare>(prepare)](data *d) mutable {
                std::invoke(prepare, d);
            });
        post_factorization_correction_step();
        graph_.for_each_parallel(
            [finalize = std::forward<Finalize>(finalize)](data *d) mutable {
                std::invoke(finalize, d);
            });
        refresh_ls_bounds();
    }
    void solve_direction(iteration_context &ctx, bool do_scaling, bool gauss_newton);
    void correct_direction(iteration_context &ctx, bool do_refinement);
    void prepare_globalization(filter_linesearch_data &ls, iteration_context &ctx);
    bool evaluate_trial_point(filter_linesearch_data &ls, iteration_context &ctx);
    void accept_trial_point(filter_linesearch_data &ls, iteration_context &ctx);
    line_search_action run_globalization(filter_linesearch_data &ls, iteration_context &ctx);
    /**
     * @brief Run one QP solve + line-search iteration.
     *
     * @param ls             Filter line-search state (updated in-place).
     * @param kkt_current    KKT info of the current point (updated to the accepted trial on return).
     * @param do_scaling     Whether to apply Jacobian scaling before factorization.
     * @param do_refinement  Whether to run iterative refinement after the corrector.
     * @param gauss_newton   If true, calls ns_factorization with gauss_newton=true.
     * @return               Line-search action that terminated the inner loop.
     */
    line_search_action sqp_iter(filter_linesearch_data &ls, kkt_info &kkt_current,
                                bool do_scaling, bool do_refinement,
                                bool gauss_newton = false);
    /**
     * @brief Bind a callback to the current @ref riccati_solver_ instance
     *
     * @tparam Func function type
     * @param f function to be bound, must have the first argument as a pointer to @ref solver_type
     * @return decltype(auto) the bound function
     * @note the function can have any number of additional arguments
     */
    template <typename Func>
    decltype(auto) solver_call(Func f) {
        using arg_type = utils::func_traits<decltype(f)>::arg_types;
        auto make_wrapper = [this, f]<typename... Args>(std::tuple<Args...> *) {
            return [this, f](Args... args) {
                std::invoke(f, riccati_solver_, std::forward<Args>(args)...);
            };
        };
        return make_wrapper((arg_type *)nullptr);
    }
};

} // namespace moto

#endif
