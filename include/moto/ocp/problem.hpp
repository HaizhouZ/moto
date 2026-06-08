#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <memory>
#include <string>
#include <type_traits>
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
class node_ocp;
def_ptr(node_ocp);
class edge_ocp;
def_ptr(edge_ocp);
class graph_model;

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
    bool add_impl(shared_expr, bool terminal = false);
    bool add_terminal_impl(expr &);
    void maintain_order();
    bool finalized_ = false;
    utils::unique_id<ocp_base> uid_;
    std::array<expr_list, field::num> disabled_expr_, pruned_expr_;
    std::unordered_set<size_t> uids_, disabled_uids_, pruned_uids_;

    void set_dim_and_idx();
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

    size_t get_expr_start(const expr &ex) const;
    size_t get_expr_start_tangent(const expr &ex) const;

    virtual bool accepts_term(const shared_expr &ex, bool terminal = false, std::string *reason = nullptr) const;
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

class node_ocp : public ocp {
  protected:
    node_ocp() = default;
    node_ocp(const node_ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<node_ocp>(new node_ocp()); }
    node_ocp_ptr_t clone_node(const active_status_config &config = {}) const;
    bool accepts_term(const shared_expr &ex, bool terminal = false, std::string *reason = nullptr) const override;

};

class edge_ocp : public ocp {
    friend class graph_model;

  protected:
    edge_ocp() = default;
    edge_ocp(const edge_ocp &rhs) = default;

  public:
    static auto create() { return std::shared_ptr<edge_ocp>(new edge_ocp()); }

  private:
    edge_ocp_ptr_t clone_edge(const active_status_config &config = {}) const;

    node_ocp_ptr_t st_node_prob_;
    node_ocp_ptr_t ed_node_prob_;

    void bind_nodes(const node_ocp_ptr_t &st, const node_ocp_ptr_t &ed = {});
    const node_ocp_ptr_t &st_node_prob() const { return st_node_prob_; }
    const node_ocp_ptr_t &ed_node_prob() const { return ed_node_prob_; }
};

} // namespace moto

extern template void moto::ocp_base::add<const moto::shared_expr &>(const moto::shared_expr &ex);
extern template void moto::ocp_base::add<const moto::shared_expr>(const moto::shared_expr &&ex);
extern template void moto::ocp_base::add<moto::shared_expr>(moto::shared_expr &&ex);

#endif // __MOTO_PROBLEM_HPP__
