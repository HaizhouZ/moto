#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <moto/core/expr.hpp>
#include <moto/ocp/sym.hpp>

namespace moto {
class ocp_base;
def_ptr(ocp_base);
class ocp;
def_ptr(ocp);
class node_ocp;
def_ptr(node_ocp);
class edge_ocp;
def_ptr(edge_ocp);

/// Base container for a stage/transition problem.
///
/// `ocp_base` owns the expression lists, activation/pruning state, and the
/// flattened indexing metadata used by solver-side data structures. The
/// derived types only add modeling semantics:
/// - `ocp`: generic problem container used by the legacy API
/// - `node_ocp`: node-wise problem, intended for `(x)` / `(x, u)` terms
/// - `edge_ocp`: edge-wise problem, intended for transition terms that bind
///   two nodes together
class ocp_base {
  public:
    struct active_status_config {
        expr_inarg_list deactivate_list;
        expr_inarg_list activate_list;
        bool empty() const { return deactivate_list.empty() && activate_list.empty(); }
    };

  protected:
    ocp_base() { uid_.set_inc(); }
    ocp_base(const ocp_base &rhs) = default;
    bool add_impl(expr &);
    bool add_impl(shared_expr, bool terminal = false);
    bool add_terminal_impl(expr &);
    void maintain_order();
    static size_t max_uid;
    bool finalized_ = false;
    utils::unique_id<ocp_base> uid_;
    std::array<expr_list, field::num> expr_, disabled_expr_, pruned_expr_;
    std::unordered_map<size_t, size_t> flatten_idx_;
    std::unordered_map<size_t, size_t> flatten_tidx_;
    std::unordered_map<size_t, size_t> pos_by_uid_;
    std::array<size_t, field::num> dim_{};
    std::array<size_t, field::num_prim> tdim_{};
    std::unordered_set<size_t> uids_, disabled_uids_, pruned_uids_;

    void set_dim_and_idx();
    void finalize();
    /// Refresh clone-local caches and recursively clone sub-problems after a
    /// copy-construction based clone.
    void refresh_after_clone(const active_status_config &config);
    inline void field_read_guard() const {
        assert(finalized_ && "Cannot access before the problem is finalized. Please call finalize() before accessing expressions.");
    }
    scalar_t *get_data_ptr(scalar_t *data, const expr &ex) const {
        return data + get_expr_start(ex);
    }

  public:
    CONST_PROPERTY(uid);
    /// Expressions active in a given field of the current problem.
    const auto &exprs(size_t f) const { return expr_.at(f); }
    /// Position of an expression inside its field-local storage.
    const auto &pos(const expr &ex) const {
        field_read_guard();
        return pos_by_uid_.at(ex.uid());
    }
    /// Flattened dimension of a field in the current problem.
    size_t dim(size_t f) const {
        field_read_guard();
        return dim_.at(f);
    }
    /// Number of active expressions in a field.
    size_t num(size_t f) const { return expr_[f].size(); }
    /// Flattened tangent-space dimension of a primal field.
    size_t tdim(size_t f) const {
        field_read_guard();
        return tdim_.at(f);
    }
    /// Whether an expression is present in this problem.
    /// If `include_sub_prob` is true, nested sub-problems are searched too.
    bool contains(const expr &ex, bool include_sub_prob = true) const;
    /// Whether an expression is currently active in this problem.
    bool is_active(const expr &ex, bool include_sub_prob = true) const;
    /// Wait for all expressions to be ready, then finalize the problem.
    void wait_until_ready();
    /// Print a compact summary grouped by field.
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

    /// Add an expression with terminal semantics.
    ///
    /// This keeps the user-facing handle reusable while allowing the problem to
    /// lower the terminal insertion independently when needed.
    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add_terminal(T &&ex) { add_terminal_impl(ex); }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }
    void add_terminal(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add_terminal(ex);
        }
    }

    void add(const ocp_base_ptr_t &sub) {
        sub_probs_.push_back(sub);
    }

    /// Start offset of an expression in the flattened storage of its field.
    size_t get_expr_start(const expr &ex) const {
        try {
            field_read_guard();
            return flatten_idx_.at(ex.uid());
        } catch (const std::exception &) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }

    /// Start offset of an expression in the flattened tangent storage.
    size_t get_expr_start_tangent(const expr &ex) const {
        try {
            field_read_guard();
            return flatten_tidx_.at(ex.uid());
        } catch (const std::exception &) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }

    /// Clone the concrete problem type, optionally overriding active status.
    virtual ocp_base_ptr_t clone_base(const active_status_config &config = {}) const = 0;
    /// Hook for future node/edge validation. The default implementation accepts
    /// every term.
    virtual bool accepts_term(const shared_expr &ex, bool terminal = false, std::string *reason = nullptr) const;
    /// Apply activation/deactivation rules and lazy pruning.
    void update_active_status(const active_status_config &config, bool update_sub_probs = true);

  protected:
    bool allow_inconsistent_dynamics_ = false;
    bool automatic_reorder_primal_ = true;
    std::vector<ocp_base_ptr_t> sub_probs_;

  public:
    PROPERTY(allow_inconsistent_dynamics)
    PROPERTY(automatic_reorder_primal)
    const auto &sub_probs() const { return sub_probs_; }
};

class ocp : public ocp_base {
  protected:
    ocp() = default;
    ocp(const ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<ocp>(new ocp()); }
    /// Clone the generic problem container.
    ocp_ptr_t clone(const active_status_config &config = {}) const;
    ocp_base_ptr_t clone_base(const active_status_config &config = {}) const override {
        return clone(config);
    }
};

/// Node-wise problem wrapper.
///
/// This is intentionally thin today: it reuses all storage/finalization logic
/// from `ocp_base`, but gives the modeling layer a dedicated type to represent
/// node-local terms and terminal composition.
class node_ocp : public ocp {
  protected:
    node_ocp() = default;
    node_ocp(const node_ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<node_ocp>(new node_ocp()); }
    /// Build a composed node problem from a base node problem.
    static node_ocp_ptr_t compose(const node_ocp_ptr_t &base_prob);
    /// Clone a node problem while preserving the concrete type.
    node_ocp_ptr_t clone_node(const active_status_config &config = {}) const;
    ocp_base_ptr_t clone_base(const active_status_config &config = {}) const override {
        return clone_node(config);
    }
};

/// Edge-wise problem wrapper.
///
/// An `edge_ocp` may bind a start/end node pair and then compose the start-node
/// terms together with edge-local terms into the transition problem consumed by
/// the solver.
class edge_ocp : public ocp {
  protected:
    edge_ocp() = default;
    edge_ocp(const edge_ocp &rhs) = default;
    node_ocp_ptr_t st_node_prob_;
    node_ocp_ptr_t ed_node_prob_;

  public:
    static auto create() { return std::shared_ptr<edge_ocp>(new edge_ocp()); }
    /// Compose an edge problem by injecting the start-node problem into it.
    static edge_ocp_ptr_t compose(const node_ocp_ptr_t &st_node_prob,
                                  const edge_ocp_ptr_t &edge_prob,
                                  const node_ocp_ptr_t &lowered_node_prob = {},
                                  bool skip_st_path_state_terms = false);
    /// Clone an edge problem while preserving the concrete type.
    edge_ocp_ptr_t clone_edge(const active_status_config &config = {}) const;
    /// Bind the start/end node problems referenced by this edge.
    void bind_nodes(const node_ocp_ptr_t &st, const node_ocp_ptr_t &ed = {});
    /// Compose using the currently bound nodes.
    edge_ocp_ptr_t compose() const;
    const node_ocp_ptr_t &st_node_prob() const { return st_node_prob_; }
    const node_ocp_ptr_t &ed_node_prob() const { return ed_node_prob_; }
    ocp_base_ptr_t clone_base(const active_status_config &config = {}) const override {
        return clone_edge(config);
    }
};

} // namespace moto

extern template void moto::ocp_base::add<const moto::shared_expr &>(const moto::shared_expr &ex);
extern template void moto::ocp_base::add<const moto::shared_expr>(const moto::shared_expr &&ex);
extern template void moto::ocp_base::add<moto::shared_expr>(moto::shared_expr &&ex);

#endif // __MOTO_PROBLEM_HPP__
