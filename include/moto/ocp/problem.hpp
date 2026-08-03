#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <moto/core/expr.hpp>
#include <moto/core/field_layout_store.hpp>
#include <moto/core/sparse.hpp>
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
class graph_composer;

enum class stage_expr_role : size_t {
    interval,
    start_node,
    end_node,
};

enum class linear_target : size_t { jacobian, lag_hessian, hessian_modification };

struct ocp_linear_profile {
    std::unordered_map<size_t, std::vector<sparse_block_spec>> blocks;
    static constexpr size_t key(linear_target target, field_t a, field_t b) {
        return (static_cast<size_t>(target) * field::num + a) * field::num + b;
    }
    const std::vector<sparse_block_spec> &get(linear_target target, field_t a, field_t b) const {
        static const std::vector<sparse_block_spec> empty;
        const auto it = blocks.find(key(target, a, b));
        return it == blocks.end() ? empty : it->second;
    }
};

class ocp_base : protected field_layout_store<expr_list> {
  public:
    struct active_status_config {
        expr_list deactivate_list;
        expr_list activate_list;
        active_status_config() = default;
        active_status_config(const expr_inarg_list &deactivate,
                             const expr_inarg_list &activate)
            : deactivate_list(deactivate), activate_list(activate) {}
        bool empty() const { return deactivate_list.empty() && activate_list.empty(); }
    };

  protected:
    ocp_base();
    ocp_base(const ocp_base &rhs);
    ~ocp_base();
    bool add_impl(expr_handle);
    void maintain_order();
    virtual void on_modified();
    bool finalized_ = false;
    utils::unique_id<ocp_base> uid_;
    std::array<expr_list, field::num> disabled_expr_, pruned_expr_;
    std::unordered_set<size_t> uids_, disabled_uids_, pruned_uids_;
    ocp_linear_profile linear_profile_;

    void finalize();
    void build_linear_profile();
    void refresh_copy(const active_status_config &config);
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
    const ocp_linear_profile &linear_profile() const {
        field_read_guard();
        return linear_profile_;
    }

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

    void add(expr_handle ex) { add_impl(std::move(ex)); }
    void add(expr &ex) { add(ex.handle()); }
    void add(const expr &ex) { add(ex.handle()); }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    size_t get_expr_start(const expr &ex) const;
    size_t get_expr_start_tangent(const expr &ex) const;

    virtual bool accepts_term(const expr_handle &ex, std::string *reason = nullptr) const;
    void update_active_status(const active_status_config &config);

  protected:
    bool allow_inconsistent_dynamics_ = false;
    bool automatic_reorder_primal_ = true;

  public:
    bool allow_inconsistent_dynamics() const { return allow_inconsistent_dynamics_; }
    void set_allow_inconsistent_dynamics(bool value) {
        if (allow_inconsistent_dynamics_ != value) {
            allow_inconsistent_dynamics_ = value;
            on_modified();
        }
    }
    bool automatic_reorder_primal() const { return automatic_reorder_primal_; }
    void set_automatic_reorder_primal(bool value) {
        if (automatic_reorder_primal_ != value) {
            automatic_reorder_primal_ = value;
            on_modified();
        }
    }
};

class ocp : public ocp_base {
  protected:
    ocp() = default;
    ocp(const ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<ocp>(new ocp()); }
    ocp_ptr_t copy(const active_status_config &config = {}) const;

  protected:
};

class stage_ocp : public ocp, public std::enable_shared_from_this<stage_ocp> {
    friend class node_view;
    friend class graph_model;
    friend class graph_composer;

  protected:
    stage_ocp() = default;
    stage_ocp(const stage_ocp &rhs);

  private:
    std::unordered_map<size_t, unsigned> endpoint_role_mask_by_uid_;
    std::function<void()> mutation_callback_;
    std::atomic<size_t> mutation_revision_{1};
    bool add_with_role(expr_handle ex, stage_expr_role role);
    bool validate_stage_term(const expr_handle &ex, std::string *reason) const;
    bool validate_endpoint_term(const expr_handle &ex, std::string *reason) const;
    void set_mutation_callback(std::function<void()> callback);
    size_t mutation_revision() const noexcept {
        return mutation_revision_.load(std::memory_order_acquire);
    }
    bool has_role(const expr &ex, stage_expr_role role) const;
    void on_modified() override;

  public:
    static auto create() { return std::shared_ptr<stage_ocp>(new stage_ocp()); }
    /// Independent stage container sharing immutable expression handles.
    stage_ocp_ptr_t copy(const active_status_config &config = {}) const;
    bool accepts_term(const expr_handle &ex, std::string *reason = nullptr) const override;

    void add(expr_handle ex) { add_with_role(std::move(ex), stage_expr_role::interval); }
    void add(expr &ex) { add(ex.handle()); }
    void add(const expr &ex) { add(ex.handle()); }

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
    void add(expr_handle ex) {
        auto owner = owner_;
        if (!owner) {
            throw std::runtime_error("Cannot add to an empty endpoint");
        }
        if (!ex) {
            throw std::runtime_error("Cannot add null expression to endpoint");
        }
        std::string reason;
        if (!owner->validate_endpoint_term(ex, &reason)) {
            throw std::runtime_error(fmt::format(
                "Cannot add expression {} uid {} to endpoint: {}",
                ex->name(), ex->uid(), reason));
        }
        owner->add_with_role(std::move(ex), role_);
    }
    void add(expr &ex) { add(ex.handle()); }
    void add(const expr &ex) { add(ex.handle()); }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    stage_ocp_ptr_t stage() const { return owner_; }
    stage_expr_role role() const { return role_; }
    explicit operator bool() const { return bool(owner_); }

  private:
    stage_ocp_ptr_t owner_;
    stage_expr_role role_ = stage_expr_role::start_node;
};

} // namespace moto

#endif // __MOTO_PROBLEM_HPP__
