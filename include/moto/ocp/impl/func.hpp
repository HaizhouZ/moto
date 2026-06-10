#ifndef __MOTO_OCP_IMPL_FUNC_HPP__
#define __MOTO_OCP_IMPL_FUNC_HPP__

#include <moto/core/field_layout_store.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/utils/movable_ptr.hpp>
#include <map>
#include <memory>
#include <string_view>
#include <utility>

namespace moto {
class func_codegen;
class generic_func;
class graph_model;
namespace utils {
namespace cs_codegen {
struct task;
}
} // namespace utils
using func = utils::shared<generic_func>; ///< shared pointer type for generic_func
class generic_func : public expr, protected field_layout_store<var_list> {
  public:
    using symbol_remap = std::vector<std::pair<var, var>>;

  protected:
    using remap_key = std::vector<std::pair<size_t, size_t>>;
    struct normalized_remap {
        std::map<size_t, std::pair<var, var>> entries;
        remap_key key;
        bool empty() const noexcept { return key.empty(); }
    };
    struct gen_info {
        using task_type = utils::cs_codegen::task;
        movable_ptr<task_type> task_ = nullptr;
        gen_info() = default;
        gen_info(const gen_info &rhs);
        gen_info(gen_info &&) = default;
        gen_info &operator=(const gen_info &rhs);
        gen_info &operator=(gen_info &&) = default;
        ~gen_info();
    };
    gen_info gen_;
    bool zero_dim_ = false;
    approx_order order_ = approx_order::first;
    var_list in_args_;
    expr_list enable_if_all_deps_;
    expr_list disable_if_any_deps_;
    expr_list enable_if_any_deps_;

    struct remap_cache;
    std::unique_ptr<remap_cache> remap_cache_;

    std::set<size_t> skip_unused_arg_check_;
    std::vector<sp_info> jac_sp_;
    std::vector<std::vector<sp_info>> hess_sp_;

    sparsity default_hess_sp_ = sparsity::dense;
    bool detect_jacobian_sparsity_ = true;

    friend class func_arg_map;
    friend class func_approx_data;
    friend class graph_model;

    virtual void substitute(const sym &arg, const sym &rhs);
    void substitute_argument(const sym &arg, const sym &rhs);
    void set_from_casadi(const var_inarg_list &in_args, const cs::SX &out);
    virtual void setup_hess();
    void disable_jacobian_sparsity_detection() {
        field_write_guard();
        detect_jacobian_sparsity_ = false;
    }
    void enable_jacobian_sparsity_detection() {
        field_write_guard();
        detect_jacobian_sparsity_ = true;
    }
    bool detect_jacobian_sparsity() const { return detect_jacobian_sparsity_; }

    virtual void finalize_impl() override;
    virtual void value_impl(func_approx_data &data) const;
    virtual void jacobian_impl(func_approx_data &data) const;
    virtual void hessian_impl(func_approx_data &data) const;
    virtual void load_external_impl(const std::string &path = "gen");
    void rebuild_argument_layout();
    normalized_remap normalize_argument_remap(const symbol_remap &remap) const;
    void apply_argument_remap(const normalized_remap &remap,
                              std::string_view context = {},
                              size_t problem_uid = static_cast<size_t>(-1));
    shared_expr remap_arguments_cached(const symbol_remap &remap,
                                       std::string_view context = {},
                                       size_t problem_uid = static_cast<size_t>(-1));
    shared_expr lower_expr_x_to_y_cached(std::string_view context = {},
                                         size_t problem_uid = static_cast<size_t>(-1));

    generic_func();
    generic_func(const generic_func &);
    generic_func &operator=(const generic_func &) = delete;

    void field_write_guard(field_t field = __undefined) const {
        assert(!finalized_ && "cannot modify function after finalized");
    }

    void field_read_guard(field_t field) const {
        assert(finalized_ && "function not finalized");
        assert(field < field::num && "field out of range");
    }

  public:
    generic_func(const std::string &name, approx_order order, size_t dim, field_t field = __undefined);
    generic_func(const std::string &name, const var_inarg_list &in_args, const cs::SX &out,
                 approx_order order, field_t field = __undefined);

    generic_func(generic_func &&) noexcept;
    generic_func &operator=(generic_func &&) noexcept = delete;
    ~generic_func() override;

    const auto &order() const { return order_; }
    const auto &__get_order() const { return order_; }
    const auto &in_args() const { return in_args_; }
    const auto &in_args(size_t i) const { return in_args_[i]; }

    const auto &jac_sparsity() const { return jac_sp_; }
    const auto &hess_sparsity() const { return hess_sp_; }

    void set_jac_sparsity(const sym &arg, sp_info sp) {
        field_write_guard(arg.field());
        jac_sp_.push_back(sp);
    }
    void set_jac_sparsity(const sym &arg, sparsity sp) {
        set_jac_sparsity(arg, {sp, 0, 0});
    }
    void set_hess_sparsity(const std::vector<std::vector<sp_info>> &sp) {
        field_write_guard();
        hess_sp_ = sp;
    }
    void set_default_hess_sparsity(sparsity sp) {
        field_write_guard();
        default_hess_sp_ = sp;
    }

    const var_list &in_args(field_t field) const;
    size_t arg_num(field_t field) const;
    size_t arg_dim(field_t field) const;
    size_t arg_tdim(field_t field) const;
    bool has_arg(const sym &s) const;
    size_t arg_idx(const sym &s) const;
    void add_argument(const sym &in);
    void add_argument(const var &in);
    void add_arguments(const var_inarg_list &args);

    const bool check_enable(ocp_base *prob) const;
    void enable_if_all(const expr_inarg_list &args);
    void disable_if_any(const expr_inarg_list &args);
    void enable_if_any(const expr_inarg_list &args);

    virtual func_approx_data_ptr_t create_approx_data(sym_data &primal,
                                                      lag_data &raw,
                                                      shared_data &shared) const;
    void compute_approx(func_approx_data &data,
                        bool eval_val, bool eval_jac = false, bool eval_hess = false) const;

    auto *get_codegen_task() { return gen_.task_.get(); }
    const auto *get_codegen_task() const { return gen_.task_.get(); }

    void load_external(const std::string &path = "gen");
    std::function<void(func_approx_data &)> value;
    std::function<void(func_approx_data &)> jacobian;
    std::function<void(func_approx_data &)> hessian;

    DEF_DEFAULT_CLONE(generic_func)

    bool has_u_arg() const;
    bool has_pure_x_primal_args() const;
    shared_expr remap_arguments(const symbol_remap &remap);
};

} // namespace moto

#endif /*__MOTO_OCP_IMPL_FUNC_HPP__*/
