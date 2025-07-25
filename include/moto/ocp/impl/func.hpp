#ifndef __approx__
#define __approx__

#include <future>
#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func_data.hpp>

namespace moto {
class func_codegen;
using sym_list = std::vector<sym>; ///< initializer list of symbolic expressions
/**
 * @brief approximation class for generic functions
 * @todo: change to differentiable for precompute
 */
class func : public expr {
  public:
    struct impl : public expr::impl {
        /**
         * @brief codegen info
         */
        struct gen_info {
            cs::SX out_;
            std::future<void> res_;
            gen_info() = default; ///< default constructor
            gen_info(gen_info &&) = default;
            gen_info &operator=(const gen_info &rhs) {
                out_ = rhs.out_;
                return *this;
            }
        } gen_; /// < code generation info

        approx_order order_; ///< order of approximation, default is first order
        sym_list in_args_;   ///< input arguments

        std::unordered_map<size_t, size_t> sym_uid_idx_; /// < map from sym uid to index in in_args_

        /// @brief callback to evaluate the value of the approximation
        std::function<void(func_approx_map &)> value_;
        /// @brief callback to evaluate the jacobian of the approximation
        std::function<void(func_approx_map &)> jacobian_;
        /// @brief callback to evaluate the hessian of the approximation
        std::function<void(func_approx_map &)> hessian_;

        impl() = default;           ///< default constructor
        impl(impl &&rhs) = default; ///< move constructor
        explicit impl(expr::impl &&rhs)
            : expr::impl(std::move(rhs)), order_(approx_order::first) {}
        impl &operator=(const impl &rhs) = default; ///< copy assignment operator
        /**
         * @brief substitute the input argument with another symbol
         *
         * @param arg symbol to be substituted
         * @param rhs symbol to substitute with
         */
        void substitute(const sym &arg, const sym &rhs);
    };
    friend class func_codegen; ///< allow func_codegen to access private members

    virtual void load_external_impl(const std::string &path = "gen");

    DEF_IMPL_GETTER();

    /**
     * @brief finalize the function
     * @details it will wait until the codegen is done if set_from_casadi is used
     */
    virtual void finalize_impl();
    ///@brief callback to evaluate (the residual) of the function, call to value_ by default
    virtual void value_impl([[maybe_unused]] func_approx_map &data) const { get_impl().value_(data); };
    ///@brief callback to evaluate jacobian of the function, call to jacobian_ by default
    virtual void jacobian_impl([[maybe_unused]] func_approx_map &data) const { get_impl().jacobian_(data); };
    ///@brief callback to evaluate hessian of the function, call to hessian_ by default
    virtual void hessian_impl([[maybe_unused]] func_approx_map &data) const { get_impl().hessian_(data); };

  public:
    template <typename T>
        requires std::is_same_v<sym, std::remove_cvref_t<T>>
    void add_argument(T &&in) {
        auto &_in_args = get_impl().in_args_;
        _in_args.emplace_back(std::forward<T>(in));
        add_dep(_in_args.back());
        get_impl().sym_uid_idx_[_in_args.back().uid()] = get_impl().sym_uid_idx_.size();
    }

    void add_arguments(const sym_list &args) {
        for (auto &in : args) {
            add_argument(in);
        }
    }

    /**
     * @brief generate the function from casadi expression
     *
     * @param in_args input arguments, must be a list of sym
     * @param out output casadi SX expression
     */
    void set_from_casadi(const sym_list& in_args, const cs::SX &out);

    /// @brief get input argument values
    IMPL_ATTR_GETTER(in_args, func);
    /// @brief get input argument by index
    const auto &in_args(size_t i) const { return get_impl().in_args_[i]; }
    /// @brief get approximation order
    IMPL_ATTR_GETTER(order, func);

    IMPL_ATTR_GETTER(sym_uid_idx, func); ///< getter for sym_uid_idx
    /// @brief order of approximation
    /**
     * @brief set up workspace data for the function
     * @details user can override this to setup the workspace data (usually by pointer) for the function data
     * @param data func_arg_map to be setup
     * @param ws_data workspace_data pointer to the settings, can be nullptr if not needed
     */
    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {}
    /**
     * @brief setup the sparse func data
     * @details will setup the mapping from the dense dense_approx_data to func_approx_map
     * @return func_approx_map_ptr_t
     */
    virtual func_approx_map_ptr_t create_approx_map(sym_data &primal,
                                                    dense_approx_data &raw,
                                                    shared_data &shared) const;
    /**
     * @brief evaluate the func
     * @param data func_approx_map to be evaluated
     * @param eval_val evaluate value if true
     * @param eval_jac evaluate jacobian if true
     * @param eval_hess evaluate hessian if true
     */
    void compute_approx(func_approx_map &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) const {
        if (eval_val)
            value_impl(data);
        if (eval_jac)
            jacobian_impl(data);
        if (eval_hess)
            hessian_impl(data);
    }
    template <typename T>
    void compute_approx(T &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) const {
        compute_approx(dynamic_cast<func_approx_map &>(data), eval_val, eval_jac, eval_hess);
    }
    /**
     * @brief load the external approximation functions
     * @note will find the functions with suffix _jac, _hess
     * @param path folder of the external functions, default is "gen"
     */
    void load_external(const std::string &path = "gen") {
        load_external_impl(path);
    }

    /**
     * @brief constructor for approximation functions
     * @note the order must be specified, otherwise it will be considered as non-approximation
     * @param name name of the function
     * @param order order of the approximation
     * @param dim dimension of the function, default is 0
     * @param field field type, default is __undefined (to be finalized later, @ref finalize_impl)
     */
    func(const std::string &name, approx_order order = approx_order::first,
         size_t dim = dim_tbd, field_t field = __undefined)
        : expr(name, dim, field) {
        impl_.reset(new impl(std::move(*impl_)));
        get_impl().order_ = order;
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
    func(const std::string &name,
         const sym_list& in_args,
         const cs::SX &out,
         approx_order order = approx_order::first, field_t field = __undefined)
        : func(name, order, (size_t)out.size1(), field) {
        assert(out.size2() == 1 && "constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

    func() = default;

    IMPL_ATTR_GETTER(value, func);
    IMPL_ATTR_GETTER(jacobian, func);
    IMPL_ATTR_GETTER(hessian, func);
};
/**
 * @brief Code generation helper for functions
 *
 */
struct func_codegen {
    /**
     * @brief add a function to the list of functions to be compiled
     * @note this will be called by the func class
     * @param f function to be added
     */
    static void add(func *f);
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
    /**
     * @brief make codegen task for the function
     *
     */
    static std::future<void> make_codegen_task(func *f);

  private:
    static std::vector<func *> code_gen_funcs_;
};

} // namespace moto

#endif /*__approx_*/