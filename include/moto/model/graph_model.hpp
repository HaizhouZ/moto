#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <functional>
#include <typeindex>
#include <type_traits>
#include <utility>
#include <vector>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>

namespace moto::model {

inline bool compose_trace_enabled() {
    static const bool enabled = std::getenv("MOTO_TRACE_COMPOSE") != nullptr;
    return enabled;
}

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
    std::shared_ptr<void> realized_runtime;
    std::type_index realized_runtime_type = typeid(void);
    void reserve(size_t node_capacity, size_t edge_capacity) {
        if (node_capacity > nodes.capacity()) {
            nodes.reserve(node_capacity);
        }
        if (edge_capacity > edges.capacity()) {
            edges.reserve(edge_capacity);
        }
    }
    bool has_incoming_edge(size_t node_id, size_t exclude_edge_id = static_cast<size_t>(-1)) const;
    bool has_outgoing_edge(size_t node_id, size_t exclude_edge_id = static_cast<size_t>(-1)) const;

    template <typename Runtime, typename... Args>
    std::shared_ptr<Runtime> ensure_runtime(Args &&...args) {
        if (!realized_runtime || realized_runtime_type != typeid(Runtime)) {
            realized_runtime = std::make_shared<Runtime>(std::forward<Args>(args)...);
            realized_runtime_type = typeid(Runtime);
        }
        return std::static_pointer_cast<Runtime>(realized_runtime);
    }

    template <typename Runtime>
    std::shared_ptr<Runtime> runtime_as() const {
        if (!realized_runtime || realized_runtime_type != typeid(Runtime)) {
            return {};
        }
        return std::static_pointer_cast<Runtime>(realized_runtime);
    }

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

    static var_list vars_from_prob(const ocp_base_ptr_t &prob, field_t field) {
        if (!prob) {
            return {};
        }
        var_list out;
        out.reserve(prob->exprs(field).size());
        for (const shared_expr &expr : prob->exprs(field)) {
            const auto *s = dynamic_cast<const sym *>(expr.get());
            if (s == nullptr) {
                continue;
            }
            out.emplace_back(const_cast<sym &>(*s));
        }
        return out;
    }

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

    model_edge_ptr_t create_edge(const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        auto st = create_node();
        auto ed = create_node();
        return connect(st, ed, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           const model_node_ptr_t &ed,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        validate_node(ed);
        if (n_edges == 0) {
            throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
        }
        // A path with n_edges introduces n_edges - 1 intermediate key nodes and n_edges edges.
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

    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        if (n_edges == 0) {
            throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
        }
        auto ed = create_node();
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

    std::vector<model_edge_ptr_t> add_path(size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        if (n_edges == 0) {
            throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
        }
        auto st = create_node();
        auto ed = create_node();
        reserve(state_->nodes.size() + n_edges - 1, state_->edges.size() + n_edges);
        std::vector<model_edge_ptr_t> edges;
        edges.reserve(n_edges);
        auto prev = st;
        for (size_t i = 0; i < n_edges; ++i) {
            auto next = (i + 1 == n_edges) ? ed : create_node();
            edges.emplace_back(connect(prev, next, base_prob));
            prev = next;
        }
        return edges;
    }

    const model_node_ptr_t &node(size_t id) const { return state_->nodes.at(id); }
    const model_edge_ptr_t &edge(size_t id) const { return state_->edges.at(id); }

    node_ocp_ptr_t compose_terminal(const model_node_ptr_t &node_h) const {
        validate_node(node_h);
        auto composed = node_h->clone_node();
        ocp::active_status_config config;
        for (size_t f = 0; f < field::num; ++f) {
            for (const shared_expr &expr : composed->exprs(f)) {
                const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get());
                const auto *constr_expr = dynamic_cast<const generic_constr *>(expr.get());
                if ((cost_expr == nullptr || !cost_expr->terminal_add()) &&
                    (constr_expr == nullptr || !constr_expr->terminal_add())) {
                    continue;
                }
                const auto *func = dynamic_cast<const generic_func *>(expr.get());
                if (func == nullptr) {
                    continue;
                }
                const bool depends_on_u = std::any_of(func->in_args().begin(), func->in_args().end(), [](const sym &arg) {
                    return arg.field() == __u;
                });
                if (!depends_on_u) {
                    continue;
                }
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

    edge_ocp_ptr_t compose(const model_edge_ptr_t &edge_h) const {
        return compose_interval(edge_h);
    }

    edge_ocp_ptr_t compose_interval(const model_edge_ptr_t &edge_h) const {
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
                for (size_t f = 0; f < field::num; ++f) {
                    if (f == __dyn) {
                        continue;
                    }
                    for (const shared_expr &expr : sink_node->exprs(f)) {
                        const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get());
                        const auto *constr_expr = dynamic_cast<const generic_constr *>(expr.get());
                        const auto *func = dynamic_cast<const generic_func *>(expr.get());
                        if ((cost_expr == nullptr && constr_expr == nullptr) || func == nullptr) {
                            continue;
                        }
                        const bool terminal_term =
                            (cost_expr != nullptr && cost_expr->terminal_add()) ||
                            (constr_expr != nullptr && constr_expr->terminal_add());
                        if (terminal_term && !opts.include_terminal_sink_terms) {
                            continue;
                        }
                        bool lowerable_term = true;
                        bool needs_lower_to_y = false;
                        bool has_u = false;
                        for (const sym &arg : func->in_args()) {
                            if (arg.field() == __x) {
                                needs_lower_to_y = true;
                                continue;
                            }
                            if (arg.field() == __y || arg.field() == __p || arg.field() == __s) {
                                continue;
                            }
                            if (arg.field() == __u) {
                                has_u = true;
                            }
                            lowerable_term = false;
                            break;
                        }
                        if (terminal_term && has_u) {
                            fmt::print(stderr,
                                       "warning: terminal node term {} depends on u and cannot be lowered onto the final edge; ignoring it\n",
                                       expr->name());
                            continue;
                        }
                        if (!terminal_term && f != __cost) {
                            continue;
                        }
                        if (!lowerable_term) {
                            continue;
                        }
                        auto *ex_func = dynamic_cast<generic_func *>(expr.get());
                        if (ex_func->lowered_) {
                            composed->add(shared_expr(ex_func->lowered_));
                            continue;
                        }
                        auto lowered = expr.clone();
                        auto *lowered_func = dynamic_cast<generic_func *>(lowered.get());
                        if (lowered_func == nullptr) {
                            continue;
                        }
                        if (needs_lower_to_y) {
                            for (const sym &arg : lowered_func->in_args()) {
                                if (arg.field() == __x) {
                                    if (compose_trace_enabled()) {
                                        fmt::print("materializing sink-node term {} in composed ocp uid {}: {} -> {} (x_terminal -> incoming y)\n",
                                                   expr->name(), composed->uid(), arg.name(), arg.next()->name());
                                    }
                                    lowered_func->substitute_argument(arg, arg.next());
                                }
                            }
                        }
                        ex_func->lowered_ = lowered;
                        composed->add(lowered);
                    }
                }
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
            const bool sink_without_outgoing = !state_->has_outgoing_edge(edge_h->ed_id_, edge_h->id_);
            auto composed = compose_interval(edge_h,
                                             {
                                                 .materialize_sink_terms = sink_without_outgoing,
                                                 .include_terminal_sink_terms = false,
                                             });
            // Materialize/finalize sequentially so repeated lowered clones with the same
            // generated function name do not race in asynchronous codegen.
            composed->wait_until_ready();
            out.emplace_back(std::move(composed));
        }
        return out;
    }

    template <typename Graph, typename NodeFactory>
    void realize_into(Graph &graph, NodeFactory &&node_factory) const {
        using node_factory_result = std::invoke_result_t<NodeFactory &, ocp_ptr_t>;
        using realized_node_type = std::remove_reference_t<decltype(graph.add(std::declval<node_factory_result>()))>;
        using realized_node_ptr = realized_node_type *;
        graph.clear();
        const size_t num_nodes = state_->nodes.size();
        const size_t num_edges = state_->edges.size();
        std::vector<realized_node_ptr> realized_nodes_by_edge(num_edges, nullptr);
        std::vector<std::vector<size_t>> incoming_edge_ids(num_nodes);
        std::vector<std::vector<size_t>> outgoing_edge_ids(num_nodes);
        std::vector<size_t> incoming_edge_count(num_nodes, 0);
        std::vector<size_t> outgoing_edge_count(num_nodes, 0);
        std::vector<size_t> edge_source_node_ids(num_edges);
        std::vector<size_t> edge_sink_node_ids(num_edges);
        std::vector<bool> sink_without_outgoing_by_edge(num_edges, false);

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            const auto &edge_h = state_->edges.at(edge_id);
            validate_edge(edge_h);
            const size_t st_node_id = edge_h->st_id();
            const size_t ed_node_id = edge_h->ed_id();
            ++incoming_edge_count.at(ed_node_id);
            ++outgoing_edge_count.at(st_node_id);
            edge_source_node_ids.at(edge_id) = st_node_id;
            edge_sink_node_ids.at(edge_id) = ed_node_id;
        }

        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            incoming_edge_ids.at(node_id).reserve(incoming_edge_count.at(node_id));
            outgoing_edge_ids.at(node_id).reserve(outgoing_edge_count.at(node_id));
        }

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            incoming_edge_ids.at(edge_sink_node_ids.at(edge_id)).push_back(edge_id);
            outgoing_edge_ids.at(edge_source_node_ids.at(edge_id)).push_back(edge_id);
        }

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            sink_without_outgoing_by_edge.at(edge_id) = outgoing_edge_ids.at(edge_sink_node_ids.at(edge_id)).empty();
        }

        for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            const auto &edge_h = state_->edges.at(edge_id);
            const bool sink_without_outgoing = sink_without_outgoing_by_edge.at(edge_id);
            auto formulation = compose_interval(edge_h,
                                               {
                                                   .materialize_sink_terms = sink_without_outgoing,
                                                   .include_terminal_sink_terms = sink_without_outgoing,
                                               });
            auto &realized_node = graph.add(std::invoke(std::forward<NodeFactory>(node_factory),
                                                        std::static_pointer_cast<ocp>(formulation)));
            realized_nodes_by_edge[edge_id] = &realized_node;
        }

        std::vector<realized_node_ptr> head_candidates;
        std::vector<realized_node_ptr> tail_candidates;
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            const auto &incoming = incoming_edge_ids.at(node_id);
            const auto &outgoing = outgoing_edge_ids.at(node_id);
            if (incoming.empty() && outgoing.empty()) {
                continue;
            }
            if (incoming.empty()) {
                if (outgoing.size() != 1) {
                    throw std::runtime_error("graph_model::realize_into expects a unique outgoing edge from the source model node");
                }
                head_candidates.push_back(realized_nodes_by_edge.at(outgoing.front()));
                continue;
            }
            if (outgoing.empty()) {
                if (incoming.size() != 1) {
                    throw std::runtime_error("graph_model::realize_into expects a unique incoming edge for a sink model node");
                }
                tail_candidates.push_back(realized_nodes_by_edge.at(incoming.front()));
                continue;
            }
            for (const size_t incoming_edge_id : incoming) {
                for (const size_t outgoing_edge_id : outgoing) {
                    graph.connect(*realized_nodes_by_edge.at(incoming_edge_id),
                                  *realized_nodes_by_edge.at(outgoing_edge_id),
                                  {2, true, true});
                }
            }
        }

        if (head_candidates.size() != 1) {
            throw std::runtime_error("graph_model::realize_into expects a single source path");
        }
        if (tail_candidates.size() != 1) {
            throw std::runtime_error("graph_model::realize_into expects a single sink path");
        }
        graph.set_head(*head_candidates.front());
        graph.set_tail(*tail_candidates.front());
        state_->dirty = false;
    }

    size_t num_nodes() const noexcept { return state_->nodes.size(); }
    size_t num_edges() const noexcept { return state_->edges.size(); }
    const std::shared_ptr<graph_model_state> &state_ptr() const noexcept { return state_; }

  private:
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
