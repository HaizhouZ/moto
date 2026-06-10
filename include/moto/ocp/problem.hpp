#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <moto/core/expr.hpp>
#include <moto/core/field_layout_store.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {
class ocp_base;
def_ptr(ocp_base);
class ocp;
def_ptr(ocp);
class stage_ocp;
def_ptr(stage_ocp);
class node_view;
class graph_model;

enum class stage_expr_role : size_t {
    interval,
    start_node,
    end_node,
};

class ocp_base : protected field_layout_store<expr_list> {
  public:
    struct active_status_config {
        expr_inarg_list deactivate_list;
        expr_inarg_list activate_list;
        bool empty() const { return deactivate_list.empty() && activate_list.empty(); }
    };

  protected:
    ocp_base();
    ocp_base(const ocp_base &rhs);
    ~ocp_base();
    bool add_impl(expr &);
    bool add_impl(shared_expr);
    void maintain_order();
    virtual void on_modified();
    bool finalized_ = false;
    utils::unique_id<ocp_base> uid_;
    std::array<expr_list, field::num> disabled_expr_, pruned_expr_;
    std::unordered_set<size_t> uids_, disabled_uids_, pruned_uids_;

    void finalize();
    void refresh_after_clone(const active_status_config &config);
    void move_active_expr(const expr &ex, bool prune);
    bool restore_inactive_expr(const expr &ex, bool from_pruned);
    inline void field_read_guard() const {
        assert(finalized_ && "Cannot access before the problem is finalized. Please call finalize() before accessing expressions.");
    }
    scalar_t *get_data_ptr(scalar_t *data, const expr &ex) const {
        return data + get_expr_start(ex);
    }

  public:
    const auto &uid() const { return uid_; }
    const expr_list &exprs(size_t f) const;
    size_t pos(const expr &ex) const;
    size_t dim(size_t f) const;
    size_t num(size_t f) const;
    size_t tdim(size_t f) const;
    bool contains(const expr &ex) const;
    bool is_active(const expr &ex) const;
    void wait_until_ready();
    void print_summary();

    vector_ref extract(vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start(ex), ex.dim());
    }
    vector_ref extract_tangent(vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start_tangent(ex), ex.tdim());
    }
    row_vector_ref extract_row(row_vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start(ex), ex.dim());
    }
    row_vector_ref extract_row_tangent(row_vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start_tangent(ex), ex.tdim());
    }

    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add(T &&ex) { add_impl(ex); }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    size_t get_expr_start(const expr &ex) const;
    size_t get_expr_start_tangent(const expr &ex) const;

    virtual bool accepts_term(const shared_expr &ex, std::string *reason = nullptr) const;
    void update_active_status(const active_status_config &config);

  protected:
    bool allow_inconsistent_dynamics_ = false;
    bool automatic_reorder_primal_ = true;

  public:
    bool allow_inconsistent_dynamics() const { return allow_inconsistent_dynamics_; }
    void set_allow_inconsistent_dynamics(bool value) { allow_inconsistent_dynamics_ = value; }
    bool automatic_reorder_primal() const { return automatic_reorder_primal_; }
    void set_automatic_reorder_primal(bool value) { automatic_reorder_primal_ = value; }
};

class ocp : public ocp_base {
  protected:
    ocp() = default;
    ocp(const ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<ocp>(new ocp()); }
    ocp_ptr_t clone(const active_status_config &config = {}) const;

  protected:
};

class stage_ocp : public ocp, public std::enable_shared_from_this<stage_ocp> {
    friend class node_view;
    friend class graph_model;

  protected:
    stage_ocp() = default;
    stage_ocp(const stage_ocp &rhs);

  private:
    std::unordered_map<size_t, unsigned> endpoint_role_mask_by_uid_;
    std::function<void()> mutation_callback_;
    bool add_with_role(shared_expr ex, stage_expr_role role);
    bool validate_stage_term(const shared_expr &ex, std::string *reason) const;
    bool validate_endpoint_term(const shared_expr &ex, std::string *reason) const;
    void set_mutation_callback(std::function<void()> callback);
    bool has_role(const expr &ex, stage_expr_role role) const;
    void on_modified() override;

  public:
    static auto create() { return std::shared_ptr<stage_ocp>(new stage_ocp()); }
    stage_ocp_ptr_t clone(const active_status_config &config = {}) const;
    bool accepts_term(const shared_expr &ex, std::string *reason = nullptr) const override;

    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add(T &&ex) {
        add_with_role(shared_expr(ex), stage_expr_role::interval);
    }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    node_view st();
    node_view ed();
};

class node_view {
    friend class graph_model;

  public:
    node_view() = default;
    node_view(const stage_ocp_ptr_t &stage, stage_expr_role role);

  public:
    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add(T &&ex) {
        auto owner = owner_.lock();
        if (!owner) {
            throw std::runtime_error("Cannot add to an expired node_view");
        }
        shared_expr shared(ex);
        if (!shared) {
            throw std::runtime_error("Cannot add null expression to node_view");
        }
        std::string reason;
        if (!owner->validate_endpoint_term(shared, &reason)) {
            throw std::runtime_error(fmt::format(
                "Cannot add expression {} uid {} to node_view: {}",
                shared->name(), shared->uid(), reason));
        }
        owner->add_with_role(std::move(shared), role_);
    }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    stage_ocp_ptr_t stage() const { return owner_.lock(); }
    stage_expr_role role() const { return role_; }
    bool expired() const { return owner_.expired(); }

  private:
    std::weak_ptr<stage_ocp> owner_;
    stage_expr_role role_ = stage_expr_role::start_node;
};

} // namespace moto

extern template void moto::ocp_base::add<const moto::shared_expr &>(const moto::shared_expr &ex);
extern template void moto::ocp_base::add<const moto::shared_expr>(const moto::shared_expr &&ex);
extern template void moto::ocp_base::add<moto::shared_expr>(moto::shared_expr &&ex);

#endif // __MOTO_PROBLEM_HPP__
