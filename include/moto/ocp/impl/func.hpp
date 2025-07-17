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
    friend struct moto::sp_arg_map;
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
    virtual void value_impl([[maybe_unused]] sp_approx_map &data) { value(data); };
    ///@brief callback to evaluate jacobian of the function, call to @ref jacobian by default
    virtual void jacobian_impl([[maybe_unused]] sp_approx_map &data) { jacobian(data); };
    ///@brief callback to evaluate hessian of the function, call to @ref hessian by default
    virtual void hessian_impl([[maybe_unused]] sp_approx_map &data) { hessian(data); };

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
     * @brief check if the function is an approximation
     * @note funcs in __undefined will be considered as (possibly) approximation
     * @return true if field is not __pre_comp or __usr_func and field value is greater than __dyn
     */
    bool is_approx() const { return (field_ != __pre_comp && field_ != __usr_func) && field_ > __dyn; }

    virtual void setup_setting(sp_arg_map &data, workspace_data *settings) {}
    /**
     * @brief setup the sparse func data
     * @details will setup the mapping from the dense approx_storage to sp_approx_map
     * @return sp_approx_map_ptr_t
     */
    virtual sp_approx_map_ptr_t make_approx_map(sym_data &primal,
                                                approx_storage &raw,
                                                shared_data &shared);
    /**
     * @brief evaluate the func
     * @param data sp_approx_map to be evaluated
     * @param eval_val evaluate value if true
     * @param eval_jac evaluate jacobian if true
     * @param eval_hess evaluate hessian if true
     */
    void evaluate_approx(sp_approx_map &data,
                         bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        if (eval_val)
            value_impl(data);
        if (eval_jac)
            jacobian_impl(data);
        if (eval_hess)
            hessian_impl(data);
    }
    template <typename T>
    void evaluate_approx(T &data,
                         bool eval_val, bool eval_jac = false, bool eval_hess = false) {
        evaluate_approx(dynamic_cast<sp_approx_map &>(data), eval_val, eval_jac, eval_hess);
    }
    /**
     * @brief load the external approximation functions
     * @note will find the functions with suffix _jac, _hess
     * @param path folder of the external functions, default is "gen"
     */
    virtual void load_external(const std::string &path = "gen");

    /**
     * @brief constructor for non-approximation functions
     *
     * @param name name of the function
     * @param field field, must explicitly belong to the non-approximation fields
     */
    func(const std::string &name, field_t field = __undefined)
        : func(name, approx_order::none, 0, field) {
        if (is_approx()) {
            throw std::runtime_error(fmt::format("func {} field type {} not qualified as non-approx",
                                                 name_, magic_enum::enum_name(field)));
        }
        // make a default make_data
        make_data = [this](sym_data &primal, shared_data &shared) {
            return std::make_unique<sp_arg_map>(primal, shared, *this);
        };
    }
    /**
     * @brief constructor for approximation functions
     * @note the order must be specified, otherwise it will be considered as non-approximation
     * @param name name of the function
     * @param order order of the approximation
     * @param dim dimension of the function, default is 0
     * @param field field type, default is __undefined (to be finalized later, @ref finalize_impl)
     */
    func(const std::string &name, approx_order order, size_t dim = dim_tbd, field_t field = __undefined)
        : impl::expr(name, dim, field), order_(order) {
    }

  public:
    /// @brief callback to evaluate the value of the approximation
    std::function<void(sp_approx_map &)> value;
    /// @brief callback to evaluate the jacobian of the approximation
    std::function<void(sp_approx_map &)> jacobian;
    /// @brief callback to evaluate the hessian of the approximation
    std::function<void(sp_approx_map &)> hessian;
    /// @brief callback to call a non-approximation function
    std::function<void(sp_arg_map &)> call;
    /// @brief callback to make data（for non-approx) @note will not be called in @ref make_approx_map
    std::function<sp_arg_map_ptr_t(sym_data &, shared_data &)> make_data;
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