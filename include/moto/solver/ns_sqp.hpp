#ifndef __NS_SQP__
#define __NS_SQP__

#include <moto/ocp/impl/shooting_node.hpp>
#include <moto/solver/ipm/ipm_config.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/solver/ns_riccati/generic_solver.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
struct generic_solver;
}
} // namespace solver
struct ns_sqp {

    struct iterative_refinement_setting {
        bool enabled = true;           ///< whether to use iterative refinement
        size_t max_iters = 5;          ///< max refinement iterations
        scalar_t prim_res_tol = 1e-10; ///< primal residual tolerance for refinement
        scalar_t dual_res_tol = 1e-10; ///< dual residual tolerance for refinement
    };

    struct linesearch_setting : public solver::linesearch_config {
        bool enabled = true;  ///< whether to use line search
        size_t max_steps = 5; ///< max line search steps
        enum class failure_backup_strategy : size_t {
            min_step,   ///< reset to the minimum step size
            best_trial, ///< reset to the best trial so far
        } failure_backup = failure_backup_strategy::best_trial;
        scalar_t primal_gamma = 1e-4;   ///< 2-obj filter primal improvement requirement: higher-> stricter
        scalar_t dual_gamma = 1e-4;     ///< 2-obj filter dual improvement requirement: higher-> stricter
        bool enable_dual_cut = true;    ///< whether to enable the strict cut for dual residual when primal residual is small
        scalar_t dual_cut_coeff = 0.99; ///< cut threshold for dual residual: higher -> looser
        scalar_t eta = 1e-4;            ///< elasticity coefficient for the dual cut when primal residual is small, used to relax the dual cut as line search step increases
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

        double prim_tol = 1e-6; ///< primal feasibility tolerance
        double dual_tol = 1e-4; ///< dual feasibility tolerance
        double comp_tol = 1e-6; ///< complementarity feasibility tolerance

        iterative_refinement_setting rf;

        bool no_except = false;

        settings_t()
            : ls(static_cast<linesearch_setting &>(*this)), ipm(static_cast<ipm_config &>(*this)) {}

      private:
        friend class ns_sqp;
        bool verbose = true;
        size_t n_worker = MAX_THREADS; ///< number of worker threads
    } settings;

    using solver_type = solver::ns_riccati::generic_solver;
    using ns_riccati_data = solver::ns_riccati::ns_riccati_data;

    struct data : public node_data, ns_riccati_data {
        data(const ocp_ptr_t &prob)
            : node_data(prob), ns_riccati_data((node_data *)this) {}
        data(data &&rhs) = default;
        static void update_approx(data *d) {
            d->update_approximation();
        }
    };

    struct kkt_info {
        bool solved = false; // whether the problem is solved
        size_t num_iter = 0; // number of iterations
        size_t ls_steps = 0; ///< line search steps

        scalar_t objective = 0.;     // objective value
        scalar_t inf_prim_res = 0.;  // primal residual (constraint violation)
        scalar_t inf_dual_res = 0.;  // dual residual (stationary condition)
        scalar_t inf_comp_res = 0.;  // (inequality) complementarity residual
        scalar_t inf_prim_step = 0.; // infinity norm of the step
        scalar_t inf_dual_step = 0.; // infinity norm of the step
    } kkt_last;
    kkt_info update(size_t n_iter, bool verbose = true);

    ns_sqp(size_t n_jobs = MAX_THREADS);
    ns_sqp(const ns_sqp &) = delete;
    ~ns_sqp() = default;
    using node_type = impl::shooting_node<data>;

    void reset_riccati_solver(solver_type *s) {
        riccati_solver_.reset(s);
    }

    auto create_node(const ocp_ptr_t &formulation) {
        return node_type(formulation, mem_);
    }

    impl::data_mgr mem_;
    directed_graph<node_type> graph_;

  private:
    std::unique_ptr<solver_type> riccati_solver_ = nullptr;

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

    /// print statistics header
    void print_stat_header();
    /// print statistics for the current iteration
    void print_stats(int i_iter, const kkt_info &info, bool hcast_ineq);
    /// compute the kkt information of the current solution
    kkt_info compute_kkt_info();
    void iterative_refinement();
    void finalize_ls_bound_and_set_to_max();
    struct filter_linesearch_per_iter_data {
        bool recompute_approx = true;
        bool stop = false;        ///< whether to stop the line search
        bool enforce_min = false; ///< whether to enforce the minimum step size
        size_t step_cnt = 0;      ///< current line search step
        scalar_t initial_alpha_primal = 0.;
        scalar_t initial_alpha_dual = 0.;
        void reset_per_iter_data() {
            new (this) filter_linesearch_per_iter_data();
        }
    };
    struct filter_linesearch_data : public filter_linesearch_per_iter_data {
        /***** filter part *****/
        struct point {
            scalar_t prim_res = std::numeric_limits<scalar_t>::infinity();
            scalar_t dual_res = std::numeric_limits<scalar_t>::infinity();
        };
        struct trial : public point, public solver::linesearch_config {
        } best_trial;
        std::vector<point> points; ///< filter for accepting line search steps
        void update_filter(scalar_t prim_new, scalar_t dual_new, settings_t &settings);
        bool try_step(scalar_t prim_new, scalar_t dual_new, settings_t &settings, scalar_t alpha);
    };

    void filter_linesearch(filter_linesearch_data &ls, const kkt_info &kkt);
    /// initialize the solver before the first iteration or after a reset, returns the initial kkt info
    kkt_info initialize();
    void correction_step();
    void finalize_correction(data *d);
    /**
     * @brief Bind a callback to the current @ref riccati_solver_ instance
     *
     * @tparam Func function type
     * @param f function to be bound, must have the first argument as a pointer to @ref solver_type
     * @return decltype(auto) the bound function
     * @note the function can have any number of additional arguments
     */
    template <typename Func>
    decltype(auto) bind(Func f) {
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