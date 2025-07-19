#ifndef __approx__
#define __approx__

#include <future>
#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func_data.hpp>

namespace moto {
class func_codegen;
namespace impl {
/**
 * @brief approximation class for generic functions
 * @todo: change to differentiable for precompute
 */
class func : public expr {
  private:
    inline static bool gen_delegated_ = false; ///< true if the codegen is delegated to @ref moto::func_codegen
    /**
     * @brief codegen info
     */
    struct gen_info {
        cs::SX out_;
        std::future<void> res_;
    } gen_; /// < code generation info
    friend class moto::func_codegen;

  protected:
    approx_order order_;       ///< order of approximation, default is first order
    std::vector<sym> in_args_; ///< input arguments

    std::unordered_map<size_t, size_t> sym_uid_idx_; /// < map from sym uid to index in in_args_
    friend struct moto::func_arg_map;
    /**
     * @brief finalize the function
     * @details it will wait until the codegen is done if set_from_casadi is used
     */
    virtual void finalize_impl();
    /**
     * @brief substitute the input argument with another symbol
     *
     * @param arg symbol to be substituted
     * @param rhs symbol to substitute with
     */
    void substitute(const sym &arg, const sym &rhs);

  protected:
    ///@brief callback to evaluate (the residual) of the function, call to @ref value by default
    virtual void value_impl([[maybe_unused]] func_approx_map &data) { value(data); };
    ///@brief callback to evaluate jacobian of the function, call to @ref jacobian by default
    virtual void jacobian_impl([[maybe_unused]] func_approx_map &data) { jacobian(data); };
    ///@brief callback to evaluate hessian of the function, call to @ref hessian by default
    virtual void hessian_impl([[maybe_unused]] func_approx_map &data) { hessian(data); };

  public:
    void add_argument(const sym &in) {
        in_args_.push_back(in);
        dep_.push_back(in);
        sym_uid_idx_[in->uid_] = sym_uid_idx_.size();
    }
    template <typename T>
    void add_arguments(const T &args) {
        for (const auto &in : args) {
            add_argument(in);
        }
    }

    void add_arguments(std::initializer_list<sym> args) {
        for (const auto &in : args) {
            add_argument(in);
        }
    }
    /**
     * @brief generate the function from casadi expression
     *
     * @param in_args input arguments, must be a list of sym
     * @param out output casadi SX expression
     */
    void set_from_casadi(std::initializer_list<sym> in_args, const cs::SX &out);

    /// @brief get input argument values
    const auto &in_args() const { return in_args_; }
    const auto &in_args(size_t i) const { return in_args_[i]; }
    /// @brief order of approximation
    inline approx_order order() { return order_; }
    /**
     * @brief set up workspace data for the function
     * @details user can override this to setup the workspace data (usually by pointer) for the function data
     * @param data func_arg_map to be setup
     * @param ws_data workspace_data pointer to the settings, can be nullptr if not needed
     */
    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) {}
    /**
     * @brief setup the sparse func data
     * @details will setup the mapping from the dense dense_approx_data to func_approx_map
     * @return func_approx_map_ptr_t
     */
    virtual func_approx_map_ptr_t create_approx_map(sym_data &primal,
                                                    dense_approx_data &raw,
                                                    shared_data &shared);
    /**
     * @brief evaluate the func
     * @param data func_approx_map to be evaluated
     * @param eval_val evaluate value if true
     * @param eval_jac evaluate jacobian if true
     * @param eval_hess evaluate hessian if true
     */
    void compute_approx(func_approx_map &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        if (eval_val)
            value_impl(data);
        if (eval_jac)
            jacobian_impl(data);
        if (eval_hess)
            hessian_impl(data);
    }
    template <typename T>
    void compute_approx(T &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        compute_approx(dynamic_cast<func_approx_map &>(data), eval_val, eval_jac, eval_hess);
    }
    /**
     * @brief load the external approximation functions
     * @note will find the functions with suffix _jac, _hess
     * @param path folder of the external functions, default is "gen"
     */
    virtual void load_external(const std::string &path = "gen");
    /**
     * @brief constructor for approximation functions
     * @note the order must be specified, otherwise it will be considered as non-approximation
     * @param name name of the function
     * @param order order of the approximation
     * @param dim dimension of the function, default is 0
     * @param field field type, default is __undefined (to be finalized later, @ref finalize_impl)
     */
    func(const std::string &name, approx_order order = approx_order::first, size_t dim = dim_tbd, field_t field = __undefined)
        : expr(name, dim, field), order_(order) {
    }

    /**
     * @brief Construct a new constr object from casadi SX expression
     *
     * @param name  name of the constraint
     * @param in_args  input arguments
     * @param out output casadi SX expression
     * @param order approximation order
     * @param field field type, default to __undefined
     */
    func(const std::string &name, std::initializer_list<sym> in_args, const cs::SX &out,
         approx_order order = approx_order::first, field_t field = __undefined)
        : func(name, order, out.size1(), field) {
        assert(out.size2() == 1 && "constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

  public:
    /// @brief callback to evaluate the value of the approximation
    std::function<void(func_approx_map &)> value;
    /// @brief callback to evaluate the jacobian of the approximation
    std::function<void(func_approx_map &)> jacobian;
    /// @brief callback to evaluate the hessian of the approximation
    std::function<void(func_approx_map &)> hessian;
};
} // namespace impl
/**
 * @brief Code generation helper for functions
 *
 */
class func_codegen {
  private:
    inline static std::vector<impl::func *> funcs_{}; ///< list of functions to be compiled
  public:
    /**
     * @brief add a function to the list of functions to be compiled
     * @note this will be called by the func class
     * @param f function to be added
     */
    static void add(impl::func *f) { funcs_.push_back(f); }
    /**
     * @brief wait until all functions are compiled
     * @note this will block until all functions are compiled
     * @param njobs number of jobs to run in parallel, default is 4
     */
    static void wait_until_all_compiled(size_t njobs = 4);
    /**
     * @brief enable code generation delegation
     * @note this will enable the code generation delegation to the helper
     */
    static void enable();
};
} // namespace moto

#endif /*__approx_*/