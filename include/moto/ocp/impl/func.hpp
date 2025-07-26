#ifndef __approx__
#define __approx__

#include <future>
#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func_data.hpp>

namespace moto {
class func_codegen;
/**
 * @brief approximation class for generic functions
 * @todo: change to differentiable for precompute
 */
class func_base : public expr {
  public:
    // --- All members from impl are now protected in func ---
  protected:
    struct gen_info {
        cs::SX out_;
        std::future<void> res_;
        gen_info() = default;
        gen_info(gen_info &&) = default;
        gen_info(const gen_info &rhs) : out_(rhs.out_) {}
        gen_info &operator=(const gen_info &rhs) {
            out_ = rhs.out_;
            return *this;
        }
    };
    gen_info gen_;
    approx_order order_ = approx_order::first;
    var_list in_args_;
    std::unordered_map<size_t, size_t> sym_uid_idx_;

    friend class func_codegen;
    friend class func_arg_map;

    void substitute(const sym &arg, const sym &rhs);
    virtual void finalize_impl() override;
    virtual void value_impl([[maybe_unused]] func_approx_map &data) const { value(data); }
    virtual void jacobian_impl([[maybe_unused]] func_approx_map &data) const { jacobian(data); }
    virtual void hessian_impl([[maybe_unused]] func_approx_map &data) const { hessian(data); }
    virtual void load_external_impl(const std::string &path = "gen");

  public:
    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {}
    func_base() = default;
    func_base(const std::string &name, approx_order order = approx_order::first,
         size_t dim = dim_tbd, field_t field = __undefined)
        : expr(name, dim, field), order_(order) {}
    func_base(const std::string &name,
         const var_list &in_args,
         const cs::SX &out,
         approx_order order = approx_order::first, field_t field = __undefined)
        : func_base(name, order, (size_t)out.size1(), field) {
        assert(out.size2() == 1 && "constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

    func_base(const func_base &) = default;
    func_base(func_base &&) = default;
    func_base &operator=(const func_base &) = default;
    func_base &operator=(func_base &&) = default;

    PROPERTY(order)
    PROPERTY(in_args)
    const auto &in_args(size_t i) const { return in_args_[i]; }

    template <typename T>
        requires std::is_same_v<var, std::remove_cvref_t<T>> || 
                 std::is_same_v<shared_expr, std::remove_cvref_t<T>>
    void add_argument(T &&in) {
        in_args_.emplace_back(std::forward<T>(in));
        add_dep(in_args_.back());
        sym_uid_idx_[in_args_.back()->uid()] = sym_uid_idx_.size();
    }
    void add_arguments(const var_list &args) {
        for (auto &in : args) {
            add_argument(in);
        }
    }

    void set_from_casadi(const var_list &in_args, const cs::SX &out);
    virtual func_approx_map_ptr_t create_approx_map(sym_data &primal,
                                                    dense_approx_data &raw,
                                                    shared_data &shared) const;
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
    void load_external(const std::string &path = "gen") {
        load_external_impl(path);
    }
    std::function<void(func_approx_map &)> value;///< value callback
    std::function<void(func_approx_map &)> jacobian;///< jacobian callback
    std::function<void(func_approx_map &)> hessian;///< hessian callback
};
struct func : public shared_object<func_base> {
    using base = shared_object<func_base>;
    using base::shared_object;
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
    static void add(func_base *f);
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
    static std::future<void> make_codegen_task(func_base *f);

  private:
    static std::vector<func_base *> code_gen_funcs_;
};

} // namespace moto

#endif /*__approx_*/