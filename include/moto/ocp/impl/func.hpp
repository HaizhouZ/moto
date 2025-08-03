#ifndef __MOTO_OCP_IMPL_FUNC_HPP__
#define __MOTO_OCP_IMPL_FUNC_HPP__

#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func_data.hpp>

namespace moto {
class func_codegen;
class generic_func;
struct func : public shared_expr {
    using base = shared_expr;
    using base::base;
    generic_func *operator->() const; ///< convert to generic_func
    template <typename T>
        requires std::is_base_of_v<generic_func, std::remove_cvref_t<T>>
    T &as() const {
        return static_cast<T &>(*this);
    } ///< convert to derived type
};
/**
 * @brief approximation class for generic functions
 * @todo: change to differentiable for precompute
 */
class generic_func : public expr {
  public:
    // --- All members from impl are now protected in func ---
  protected:
    struct gen_info {
        cs::SX out_;
    };
    gen_info gen_;
    approx_order order_ = approx_order::first;
    var_list in_args_;
    std::unordered_map<size_t, size_t> sym_uid_idx_;

    friend class func_codegen;
    friend class func_arg_map;

    void substitute(const sym &arg, const sym &rhs);
    void set_from_casadi(const var_inarg_list &in_args, const cs::SX &out);

    virtual void finalize_impl() override;
    virtual void value_impl(func_approx_data &data) const { value(data); }
    virtual void jacobian_impl(func_approx_data &data) const { jacobian(data); }
    virtual void hessian_impl(func_approx_data &data) const { hessian(data); }
    virtual void load_external_impl(const std::string &path = "gen");

    generic_func(const generic_func &) = default;
    generic_func &operator=(const generic_func &) = default;

    friend class shared_expr; ///< allow shared_expr to access private members
    friend class func;

    using wrapper_type = func;

    generic_func() = default;
    generic_func(const std::string &name, approx_order order, size_t dim, field_t field)
        : expr(name, dim, field), order_(order) {}
    generic_func(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order, field_t field)
        : generic_func(name, order, (size_t)out.size1(), field) {
        assert(out.size2() == 1 && "generic_constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

  public:
    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {}

    generic_func(generic_func &&) = default;
    generic_func &operator=(generic_func &&) = default;

    PROPERTY(order)
    PROPERTY(in_args)
    const auto &in_args(size_t i) const { return in_args_[i]; }

    template <typename T>
        requires std::is_same_v<var, std::remove_cvref_t<T>> ||
                 std::is_same_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_same_v<sym, std::remove_cvref_t<T>>
    void add_argument(T &&in) {
        in_args_.emplace_back(std::forward<T>(in));
        add_dep(in_args_.back());
        sym_uid_idx_[in_args_.back()->uid()] = sym_uid_idx_.size();
    }
    void add_arguments(const var_inarg_list &args) {
        for (sym &in : args) {
            add_argument(in);
        }
    }

    virtual func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                                      merit_data &raw,
                                                      shared_data &shared) const;
    void compute_approx(func_approx_data &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) const;

    void load_external(const std::string &path = "gen") {
        load_external_impl(path);
    }
    std::function<void(func_approx_data &)> value;    ///< value callback
    std::function<void(func_approx_data &)> jacobian; ///< jacobian callback
    std::function<void(func_approx_data &)> hessian;  ///< hessian callback

#define DEF_FUNC_CLONE                                                                                       \
    wrapper_type clone() const {                                                                             \
        assert(uid_ != uid_max && "cannot clone a null function");                                           \
        return wrapper_type(std::shared_ptr<generic_func>(new std::remove_cvref_t<decltype(*this)>(*this))); \
    }

    DEF_FUNC_CLONE;
};
inline generic_func *func::operator->() const {
    return static_cast<generic_func *>(base::operator->());
} ///< convert to generic_func
/**
 * @brief Code generation helper for functions
 *
 */
struct func_codegen {
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
    static void make_codegen_task(generic_func *f);
};

} // namespace moto

#endif /*__MOTO_OCP_IMPL_FUNC_HPP__*/