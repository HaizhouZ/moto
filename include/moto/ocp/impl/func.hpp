#ifndef __MOTO_OCP_IMPL_FUNC_HPP__
#define __MOTO_OCP_IMPL_FUNC_HPP__

#include <moto/core/workspace_data.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/utils/movable_ptr.hpp>

namespace moto {
class func_codegen;
class generic_func;
namespace utils {
namespace cs_codegen {
struct task;
}
} // namespace utils
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
        using task_type = utils::cs_codegen::task;
        movable_ptr<task_type> task_ = nullptr;
        mutable bool copy_task = true; // if true, *task_ is copied in copy constructor
        gen_info() = default;
        gen_info(const gen_info &rhs);
        gen_info(gen_info &&) = default;
        gen_info &operator=(const gen_info &rhs);
        gen_info &operator=(gen_info &&) = default;
        ~gen_info();
    };
    gen_info gen_;
    bool zero_dim_ = false; // whether the function has zero dimension
    approx_order order_ = approx_order::first;
    var_list in_args_;
    expr_list enable_if_all_deps_;  // if all these args are active, the function is enabled
    expr_list disable_if_any_deps_; // if any of these args are active, the function is disabled
    expr_list enable_if_any_deps_;  // if any of these args are active, the function is enabled
    array_type<var_list, primal_fields> arg_by_field_{};
    array_type<size_t, primal_fields> arg_dim_{};
    array_type<size_t, primal_fields> arg_tdim_{};
    array_type<size_t, primal_fields> arg_num_{};

    struct ocpwise_info {
        array_type<var_list, primal_fields> arg_by_field_{};
        array_type<size_t, primal_fields> arg_dim_{};
        array_type<size_t, primal_fields> arg_tdim_{};
        array_type<size_t, primal_fields> arg_num_{};
    };

    mutable std::unordered_map<size_t, ocpwise_info> ocpwise_info_map_;

    std::set<size_t> skip_unused_arg_check_;     ///< set of argument uids to skip unused check
    std::vector<sparsity> jac_sp_;               ///< jacobian sparsity for each arg
    std::vector<std::vector<sparsity>> hess_sp_; ///< hessian sparsity for each pair of args
    
    sparsity default_hess_sp_ = sparsity::dense;

    std::unordered_map<size_t, size_t> sym_uid_idx_;

    friend class func_arg_map;
    friend class func_approx_data;

    void substitute(const sym &arg, const sym &rhs);
    void set_from_casadi(const var_inarg_list &in_args, const cs::SX &out);

    virtual void finalize_impl() override;
    virtual void value_impl(func_approx_data &data) const;
    virtual void jacobian_impl(func_approx_data &data) const;
    virtual void hessian_impl(func_approx_data &data) const;
    virtual void load_external_impl(const std::string &path = "gen");

    generic_func(const generic_func &) = default;
    generic_func &operator=(const generic_func &) = default;

    friend class shared_expr; ///< allow shared_expr to access private members
    friend class func;

    using wrapper_type = func;

    generic_func() = default;

    void field_access_guard(field_t field) const {
        assert(finalized_ && "function not finalized");
        assert(in_field(field, primal_fields) && "field out of range");
    } ///< guard access to field-based members

    void setup_ocpwise_info(const ocp *prob) const;

  public:
    generic_func(const std::string &name, approx_order order, size_t dim, field_t field)
        : expr(name, dim, field), order_(order) {}
    generic_func(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order, field_t field)
        : generic_func(name, order, (size_t)out.size1(), field) {
        assert(out.size2() == 1 && "generic_constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {}

    /// @brief get a shared duplicate of the function with different uid
    /// @param duplicate_args whether to copy the input arguments (with new uid) or just share the same arguments
    /// @warning make sure the func is finalized before calling this function, otherwise it will block the thread
    generic_func share(bool copy_args = true, const var_inarg_list &skip_copy_args = {}) const;

    generic_func(generic_func &&) = default;
    generic_func &operator=(generic_func &&) = default;

    PROPERTY(order)
    PROPERTY(in_args)
    const auto &in_args(size_t i) const { return in_args_[i]; }

    const auto &jac_sparsity() const { return jac_sp_; }   ///< get the jacobian sparsity patterns
    const auto &hess_sparsity() const { return hess_sp_; } ///< get the hessian sparsity patterns

    void set_default_hess_sparsity(sparsity sp) { default_hess_sp_ = sp; }

    const auto &in_args(field_t field) const {
        field_access_guard(field);
        return arg_by_field_[field];
    } ///< get the input arguments for a given field

    /// get the active arguments for a given field
    const var_list &active_args(field_t f, const ocp *prob) const;
    /// get the dim of active arguments for a given field
    size_t active_dim(field_t f, const ocp *prob) const;
    /// get the tangent dim of active arguments for a given field
    size_t active_tdim(field_t f, const ocp *prob) const;
    /// get the num of active arguments for a given field
    size_t active_num(field_t f, const ocp *prob) const;

    auto arg_num(field_t field) const {
        field_access_guard(field);
        return arg_by_field_[field].size();
    } ///< get the number of arguments for a given field

    auto arg_dim(field_t field) const {
        field_access_guard(field);
        return arg_dim_[field];
    } ///< get the dimension of the argument for a given field

    auto arg_tdim(field_t field) const {
        field_access_guard(field);
        return arg_tdim_[field];
    } ///< get the tangent dimension of the argument for a given field

    bool has_arg(const sym &s) const {
        field_access_guard(s.field());
        return sym_uid_idx_.contains(s.uid());
    } ///< check if the function has argument for a given field

    size_t arg_idx(const sym &s) const {
        field_access_guard(s.field());
        return sym_uid_idx_.at(s.uid());
    } ///< get the index of the argument for a given symbol

    /// @brief add a single argument
    /// @param in argument to add, must be one of var, sym, or shared_expr
    template <typename T>
        requires std::is_same_v<var, std::remove_cvref_t<T>> ||
                 std::is_same_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_same_v<sym, std::remove_cvref_t<T>>
    void add_argument(T &&in) {
        if (std::find(in_args_.begin(), in_args_.end(), in) != in_args_.end())
            return;
        auto &s = in_args_.emplace_back(std::forward<T>(in));
        add_dep(s);
    }
    /// @brief add multiple arguments
    /// @param args list of arguments to add
    void add_arguments(const var_inarg_list &args) {
        for (sym &in : args) {
            add_argument(in);
        }
    }

    /// get the enable_if dependencies
    const bool check_enable(ocp *prob) const;
    /// enable if all of these args are active
    void enable_if_all(const expr_inarg_list &args);
    /// disable if any of these args is active
    void disable_if_any(const expr_inarg_list &args);
    /// enable if any of these args is active
    void enable_if_any(const expr_inarg_list &args);

    virtual func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                                      merit_data &raw,
                                                      shared_data &shared) const;
    /// @brief compute the approximation of the function
    /// @param data approximation data
    /// @param eval_val whether to evaluate the value
    /// @param eval_jac whether to evaluate the jacobian
    /// @param eval_hess whether to evaluate the hessian
    void compute_approx(func_approx_data &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) const;

    auto *get_codegen_task() { return gen_.task_.get(); } ///< get the codegen task

    /// @brief load external function implementation from shared library
    /// @param path path to the shared library, default is "gen"
    void load_external(const std::string &path = "gen") {
        load_external_impl(path);
    }
    /// --- Callbacks for function evaluation ---
    /// these functions can be overridden by derived classes
    std::function<void(func_approx_data &)> value;    ///< value callback
    std::function<void(func_approx_data &)> jacobian; ///< jacobian callback
    std::function<void(func_approx_data &)> hessian;  ///< hessian callback

#define DEF_FUNC_CLONE                                                                                       \
    wrapper_type clone() const {                                                                             \
        assert(uid_.is_valid() && "cannot clone a null function");                                           \
        return wrapper_type(std::shared_ptr<generic_func>(new std::remove_cvref_t<decltype(*this)>(*this))); \
    }

    DEF_FUNC_CLONE;
};
inline generic_func *func::operator->() const {
    return static_cast<generic_func *>(base::operator->());
} ///< convert to generic_func

} // namespace moto

#endif /*__MOTO_OCP_IMPL_FUNC_HPP__*/