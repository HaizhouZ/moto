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
    array_type<var_list, primal_fields> arg_by_field_{};
    array_type<size_t, primal_fields> arg_dim_{};
    array_type<size_t, primal_fields> arg_tdim_{};
    array_type<size_t, primal_fields> arg_num_{};

    std::vector<std::vector<sparsity>> hess_sp_;

    sparsity default_hess_sp_ = sparsity::dense;

    std::unordered_map<size_t, size_t> sym_uid_idx_;

    friend class func_codegen;
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
    generic_func(const std::string &name, approx_order order, size_t dim, field_t field)
        : expr(name, dim, field), order_(order) {}
    generic_func(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order, field_t field)
        : generic_func(name, order, (size_t)out.size1(), field) {
        assert(out.size2() == 1 && "generic_constr output cols must be 1");
        set_from_casadi(in_args, out);
    }

    void field_access_guard(field_t field) const {
        assert(finalized_ && "function not finalized");
        assert(in_field(field, primal_fields) && "field out of range");
    } ///< guard access to field-based members

  public:
    virtual void setup_workspace_data(func_arg_map &data, workspace_data *ws_data) const {}

    /// @brief get a shared duplicate of the function with different uid
    /// @param duplicate_args whether to copy the input arguments (with new uid) or just share the same arguments
    generic_func share(bool copy_args = true, const var_inarg_list& skip_copy_args = {}) const;

    generic_func(generic_func &&) = default;
    generic_func &operator=(generic_func &&) = default;

    PROPERTY(order)
    PROPERTY(in_args)
    const auto &in_args(size_t i) const { return in_args_[i]; }

    void set_default_hess_sparsity(sparsity sp) { default_hess_sp_ = sp; }

    const auto &in_args(field_t field) const {
        field_access_guard(field);
        return arg_by_field_[field];
    } ///< get the input arguments for a given field

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
        return sym_uid_idx_.contains(s.uid());
    } ///< check if the function has argument for a given field

    size_t arg_idx(const sym &s) const {
        return sym_uid_idx_.at(s.uid());
    } ///< get the index of the argument for a given symbol

    template <typename T>
        requires std::is_same_v<var, std::remove_cvref_t<T>> ||
                 std::is_same_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_same_v<sym, std::remove_cvref_t<T>>
    void add_argument(T &&in) {
        auto &s = in_args_.emplace_back(std::forward<T>(in));
        add_dep(s);
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

    auto *get_codegen_task() { return gen_.task_.get(); }

    void load_external(const std::string &path = "gen") {
        load_external_impl(path);
    }
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