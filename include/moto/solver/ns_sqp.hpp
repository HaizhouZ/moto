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
    struct settings_t
        : public workspace_data_collection<solver::linesearch_config, solver::ipm_config> {
        double prim_tol = 1e-6; ///< primal feasibility tolerance
        double dual_tol = 1e-4; ///< dual feasibility tolerance
        double comp_tol = 1e-6; ///< complementarity feasibility tolerance

        size_t max_rf_iters = 5; ///< max refinement iterations
        size_t max_ls_steps = 5;     ///< max line search steps
        bool use_line_search = true; ///< whether to use line search

        scalar_t mu0 = 1.0; ///< initial barrier parameter

        bool use_mu_globalization = true;     ///< whether to use mu globalization
        bool use_iterative_refinement = true; ///< whether to use iterative refinement

        bool warm_start_ipm = false; ///< whether to warm start the IPM solver
        
        bool no_except = false;

      private:
        friend class ns_sqp;
        bool verbose = true;
        size_t n_worker = MAX_THREADS; ///< number of worker threads
    } settings;
    // using node_base = ;
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
    void print_stats(int i_iter, const kkt_info &info, bool has_ineq);
    /// compute the kkt information of the current solution
    kkt_info compute_kkt_info();
    void iterative_refinement();
    void finalize_ls_bound_and_set_to_max();

    struct ls_info {
        bool recompute_approx = true;
        bool stop = false;   ///< whether to stop the line search
        bool enforce_min = false; ///< whether to enforce the minimum step size
        size_t step_cnt = 0; ///< current line search step
        scalar_t initial_alpha_primal;
        scalar_t initial_alpha_dual;
    };

    void backtrack_linesearch(ls_info &ls, const kkt_info &kkt);
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