#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>

namespace moto::model {

class graph_model;
struct graph_model_state;

class model_node;
def_ptr(model_node);
class model_edge;
def_ptr(model_edge);

struct graph_model_state {
    std::vector<model_node_ptr_t> nodes;
    std::vector<model_edge_ptr_t> edges;
    bool dirty = true;
    // Typed runtime slot for the realized graph (e.g. directed_graph<node_type>).
    // Subclasses of graph_model_state may add their own typed runtime slots.
    std::shared_ptr<void> realized_runtime;

    void reserve(size_t node_capacity, size_t edge_capacity) {
        if (node_capacity > nodes.capacity()) nodes.reserve(node_capacity);
        if (edge_capacity > edges.capacity()) edges.reserve(edge_capacity);
    }

    template <typename Runtime, typename... Args>
    std::shared_ptr<Runtime> ensure_runtime(Args &&...args) {
        if (!realized_runtime) {
            realized_runtime = std::make_shared<Runtime>(std::forward<Args>(args)...);
        }
        return std::static_pointer_cast<Runtime>(realized_runtime);
    }

    bool has_incoming_edge(size_t node_id, size_t exclude_edge_id = static_cast<size_t>(-1)) const;
    bool has_outgoing_edge(size_t node_id, size_t exclude_edge_id = static_cast<size_t>(-1)) const;

    size_t find_node_id(const node_ocp_ptr_t &prob) const {
        if (!prob) {
            throw std::runtime_error("graph_model cannot resolve a null node problem");
        }
        const auto it = std::find_if(nodes.begin(), nodes.end(), [&](const model_node_ptr_t &node) {
            return node == prob;
        });
        if (it == nodes.end()) {
            throw std::runtime_error("graph_model cannot resolve node handle from node problem");
        }
        return static_cast<size_t>(std::distance(nodes.begin(), it));
    }
};

class model_node : public node_ocp {
  protected:
    model_node() = default;
    model_node(const node_ocp &rhs) : node_ocp(rhs) {}
    std::weak_ptr<graph_model_state> owner_;
    size_t id_ = static_cast<size_t>(-1);
    friend class graph_model;

  public:
    static model_node_ptr_t create(const node_ocp_ptr_t &base_prob = {}) {
        return base_prob ? model_node_ptr_t(new model_node(*base_prob))
                         : model_node_ptr_t(new model_node());
    }

    explicit operator bool() const noexcept { return id_ != static_cast<size_t>(-1); }
    size_t id() const noexcept { return id_; }
};

class model_edge : public edge_ocp {
  protected:
    model_edge() = default;
    model_edge(const edge_ocp &rhs) : edge_ocp(rhs) {}
    std::weak_ptr<graph_model_state> owner_;
    size_t id_ = static_cast<size_t>(-1);
    size_t st_id_ = static_cast<size_t>(-1);
    size_t ed_id_ = static_cast<size_t>(-1);
    friend class graph_model;
    friend struct graph_model_state;

  public:
    static model_edge_ptr_t create(const edge_ocp_ptr_t &base_prob = {}) {
        return base_prob ? model_edge_ptr_t(new model_edge(*base_prob))
                         : model_edge_ptr_t(new model_edge());
    }

    explicit operator bool() const noexcept { return id_ != static_cast<size_t>(-1); }
    size_t id() const noexcept { return id_; }
    size_t st_id() const noexcept { return st_id_; }
    size_t ed_id() const noexcept { return ed_id_; }

    model_node_ptr_t st_node() const {
        auto owner = owner_.lock();
        if (!owner || st_id_ >= owner->nodes.size()) {
            throw std::runtime_error("model_edge has no valid start node");
        }
        return owner->nodes.at(st_id_);
    }

    model_node_ptr_t ed_node() const {
        auto owner = owner_.lock();
        if (!owner || ed_id_ >= owner->nodes.size()) {
            throw std::runtime_error("model_edge has no valid end node");
        }
        return owner->nodes.at(ed_id_);
    }
};

class graph_model {
  public:
    struct interval_compose_options {
        ocp::active_status_config source_config;
        bool materialize_sink_terms = false;
        bool include_terminal_sink_terms = false;
    };

    graph_model() : state_(std::make_shared<graph_model_state>()) {}
    explicit graph_model(std::shared_ptr<graph_model_state> state) : state_(std::move(state)) {}

    void reserve(size_t node_capacity, size_t edge_capacity) {
        state_->reserve(node_capacity, edge_capacity);
    }

    model_node_ptr_t create_node(const node_ocp_ptr_t &base_prob = node_ocp::create()) {
        auto node = model_node::create(base_prob);
        state_->dirty = true;
        node->owner_ = state_;
        node->id_ = state_->nodes.size();
        state_->nodes.emplace_back(node);
        return node;
    }

    // Backward-compatible helper for bindings/examples: create a standalone edge
    // with auto-created source and sink nodes.
    model_edge_ptr_t create_edge(const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        auto st = create_node();
        auto ed = create_node();
        return connect(st, ed, base_prob);
    }

    model_edge_ptr_t connect(const model_node_ptr_t &st,
                             const model_node_ptr_t &ed,
                             const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        validate_node(ed);
        state_->dirty = true;
        auto edge = model_edge::create(base_prob);
        edge->owner_ = state_;
        edge->id_ = state_->edges.size();
        edge->st_id_ = st->id();
        edge->ed_id_ = ed->id();
        edge->bind_nodes(st, ed);
        state_->edges.emplace_back(edge);
        return edge;
    }

    // Add a chain of n_edges edges from st to ed (creates n_edges-1 intermediate nodes).
    // Overloads: omit ed to auto-create the sink; omit both to auto-create source and sink.
    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           const model_node_ptr_t &ed,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        validate_node(ed);
        return add_path_impl(st, ed, n_edges, base_prob);
    }

    // Convenience overloads: create endpoint nodes from node problems, then build a path.
    std::vector<model_edge_ptr_t> add_path(const node_ocp_ptr_t &st_prob,
                                           const node_ocp_ptr_t &ed_prob,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path(create_node(st_prob), create_node(ed_prob), n_edges, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(const node_ocp_ptr_t &st_prob,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path(create_node(st_prob), create_node(), n_edges, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        return add_path_impl(st, create_node(), n_edges, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path_impl(create_node(), create_node(), n_edges, base_prob);
    }

    const model_node_ptr_t &node(size_t id) const { return state_->nodes.at(id); }
    const model_edge_ptr_t &edge(size_t id) const { return state_->edges.at(id); }

    node_ocp_ptr_t compose_terminal(const model_node_ptr_t &node_h) const {
        validate_node(node_h);
        auto composed = node_h->clone_node();
        ocp::active_status_config config;
        for (size_t f = 0; f < field::num; ++f) {
            const auto ff = static_cast<field_t>(f);
            if (!in_field(ff, func_fields)) continue;
            for (const shared_expr &expr : composed->exprs(f)) {
                if (!is_terminal_expr(expr, ff)) continue;
                const bool depends_on_u = expr.as<generic_func>().has_u_arg();
                if (!depends_on_u) continue;
                fmt::print(stderr,
                           "warning: terminal node term {} depends on u and cannot be applied on a terminal x/u node; ignoring it\n",
                           expr->name());
                config.deactivate_list.emplace_back(*expr);
            }
        }
        if (!config.empty()) {
            composed->update_active_status(config, false);
        }
        composed->wait_until_ready();
        return composed;
    }

    // Python binding entry point — compose a single edge into a formulation.
    edge_ocp_ptr_t compose(const model_edge_ptr_t &edge_h) const {
        return compose_interval(edge_h, interval_compose_options{});
    }

    edge_ocp_ptr_t compose_interval(const model_edge_ptr_t &edge_h,
                                    const interval_compose_options &opts) const {
        validate_edge(edge_h);
        edge_h->wait_until_ready();
        auto st_node_prob = edge_h->st_node_prob();
        if (st_node_prob) {
            st_node_prob->wait_until_ready();
        }
        if (!opts.source_config.empty()) {
            st_node_prob = st_node_prob ? st_node_prob->clone_node(opts.source_config) : node_ocp_ptr_t{};
        }
        auto composed = edge_ocp::compose(st_node_prob, edge_h, node_ocp_ptr_t{}, false);
        if (opts.materialize_sink_terms) {
            if (const auto &sink_node = edge_h->ed_node_prob()) {
                materialize_sink_terms(*sink_node, *composed, opts.include_terminal_sink_terms);
            }
        }
        composed->wait_until_ready();
        return composed;
    }

    std::vector<edge_ocp_ptr_t> compose_all() const {
        std::vector<edge_ocp_ptr_t> out;
        out.reserve(state_->edges.size());
        for (const auto &edge_h : state_->edges) {
            validate_edge(edge_h);
            const bool sink_final = !state_->has_outgoing_edge(edge_h->ed_id_, edge_h->id_);
            auto composed = compose_interval(edge_h, {
                .materialize_sink_terms = sink_final,
                .include_terminal_sink_terms = false,
            });
            // Finalize sequentially: lowered clones share a codegen name and must not race.
            composed->wait_until_ready();
            out.emplace_back(std::move(composed));
        }
        return out;
    }

    template <typename Graph, typename NodeFactory>
    void realize_into(Graph &graph, NodeFactory &&node_factory) const {
        using realized_node_type = std::remove_reference_t<decltype(graph.add(
            std::declval<std::invoke_result_t<NodeFactory &, ocp_ptr_t>>()))>;
        using realized_node_ptr = realized_node_type *;

        graph.clear();
        const size_t num_nodes = state_->nodes.size();
        const size_t num_edges = state_->edges.size();

        // Build per-node adjacency in one pass. Edges are valid by construction from connect().
        std::vector<std::vector<size_t>> incoming(num_nodes), outgoing(num_nodes);
        for (size_t eid = 0; eid < num_edges; ++eid) {
            incoming[state_->edges[eid]->ed_id()].push_back(eid);
            outgoing[state_->edges[eid]->st_id()].push_back(eid);
        }

        // Compose and realize each edge into a graph node.
        std::vector<realized_node_ptr> realized(num_edges, nullptr);
        for (size_t eid = 0; eid < num_edges; ++eid) {
            const bool sink_final = outgoing[state_->edges[eid]->ed_id()].empty();
            auto formulation = compose_interval(state_->edges[eid], {
                .materialize_sink_terms = sink_final,
                .include_terminal_sink_terms = sink_final,
            });
            realized[eid] = &graph.add(std::invoke(std::forward<NodeFactory>(node_factory),
                                                    std::static_pointer_cast<ocp>(formulation)));
        }

        // Wire up the graph topology and find the unique head and tail.
        realized_node_ptr head = nullptr, tail = nullptr;
        for (size_t nid = 0; nid < num_nodes; ++nid) {
            const auto &inc = incoming[nid];
            const auto &out = outgoing[nid];
            if (inc.empty() && out.empty()) continue;
            if (inc.empty()) {
                if (out.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique outgoing edge from the source model node");
                if (head) throw std::runtime_error("graph_model::realize_into expects a single source path");
                head = realized[out.front()];
            } else if (out.empty()) {
                if (inc.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique incoming edge for a sink model node");
                if (tail) throw std::runtime_error("graph_model::realize_into expects a single sink path");
                tail = realized[inc.front()];
            } else {
                for (size_t in_eid : inc)
                    for (size_t out_eid : out)
                        graph.connect(*realized[in_eid], *realized[out_eid], {2, true, true});
            }
        }

        if (!head) throw std::runtime_error("graph_model::realize_into expects a single source path");
        if (!tail) throw std::runtime_error("graph_model::realize_into expects a single sink path");
        graph.set_head(*head);
        graph.set_tail(*tail);
        state_->dirty = false;
    }

    size_t num_nodes() const noexcept { return state_->nodes.size(); }
    size_t num_edges() const noexcept { return state_->edges.size(); }
    const std::shared_ptr<graph_model_state> &state_ptr() const noexcept { return state_; }

  private:
    static bool is_terminal_expr(const shared_expr &expr, field_t f) {
        if (f == __cost) {
            return expr.as<generic_cost>().terminal_add();
        }
        if (in_field(f, constr_fields)) {
            return expr.as<generic_constr>().terminal_add();
        }
        return false;
    }

    void validate_node(const model_node_ptr_t &node_h) const {
        if (!node_h || node_h->owner_.lock().get() != state_.get() || node_h->id() >= state_->nodes.size()) {
            throw std::runtime_error("invalid graph_model node handle");
        }
    }

    void validate_edge(const model_edge_ptr_t &edge_h) const {
        if (!edge_h || edge_h->owner_.lock().get() != state_.get() || edge_h->id() >= state_->edges.size()) {
            throw std::runtime_error("invalid graph_model edge handle");
        }
    }

    std::vector<model_edge_ptr_t> add_path_impl(const model_node_ptr_t &st,
                                                 const model_node_ptr_t &ed,
                                                 size_t n_edges,
                                                 const edge_ocp_ptr_t &base_prob) {
        if (n_edges == 0) {
            throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
        }
        reserve(state_->nodes.size() + n_edges - 1, state_->edges.size() + n_edges);
        std::vector<model_edge_ptr_t> edges;
        edges.reserve(n_edges);
        auto prev = st;
        for (size_t i = 0; i < n_edges; ++i) {
            auto next = (i + 1 == n_edges) ? ed : create_node(prev->clone_node());
            edges.emplace_back(connect(prev, next, base_prob));
            prev = next;
        }
        return edges;
    }

    // Lower sink-node terms onto an already-composed edge formulation.
    // Non-terminal __cost terms are lowered unconditionally.
    // Terminal terms (any field) are lowered only when include_terminal is true.
    static void materialize_sink_terms(const node_ocp &sink_node,
                                       edge_ocp &composed,
                                       bool include_terminal) {
        // Non-terminal running costs on the sink: always lowered onto the final edge.
        for (const shared_expr &expr : sink_node.exprs(__cost)) {
            if (!is_terminal_expr(expr, __cost)) {
                lower_expr_into(expr, composed);
            }
        }

        // Terminal-flagged terms: any field, only when include_terminal is set.
        if (!include_terminal) return;
        for (size_t f = 0; f < field::num; ++f) {
            const auto ff = static_cast<field_t>(f);
            if (ff == __dyn || !in_field(ff, func_fields)) continue;
            for (const shared_expr &expr : sink_node.exprs(f)) {
                if (!is_terminal_expr(expr, ff)) continue;
                // Terminal terms that depend on u cannot be lowered to the edge.
                if (expr.as<generic_func>().has_u_arg()) {
                    fmt::print(stderr, "warning: terminal node term {} depends on u and cannot be lowered onto the final edge; ignoring it\n",
                               expr->name());
                    continue;
                }
                lower_expr_into(expr, composed);
            }
        }
    }

    // Clone expr, substitute any __x args to __y (x_terminal → incoming y), cache in lowered_.
    static void lower_expr_into(const shared_expr &expr, edge_ocp &composed) {
        composed.add(expr.as<generic_func>().lower_expr_x_to_y_cached(
            fmt::format("sink-node term {} materialization", expr->name()),
            composed.uid()));
    }

    std::shared_ptr<graph_model_state> state_;
};

inline bool graph_model_state::has_incoming_edge(size_t node_id, size_t exclude_edge_id) const {
    return std::any_of(edges.begin(), edges.end(), [&](const model_edge_ptr_t &edge) {
        return edge && edge->id_ != exclude_edge_id && edge->ed_id_ == node_id;
    });
}

inline bool graph_model_state::has_outgoing_edge(size_t node_id, size_t exclude_edge_id) const {
    return std::any_of(edges.begin(), edges.end(), [&](const model_edge_ptr_t &edge) {
        return edge && edge->id_ != exclude_edge_id && edge->st_id_ == node_id;
    });
}

} // namespace moto::model

#endif
