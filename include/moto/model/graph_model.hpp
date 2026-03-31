#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <vector>

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
    node_ocp_ptr_t compose() const;
    node_ocp_ptr_t compose_terminal() const;
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
    model_node_ptr_t st() const;
    model_node_ptr_t ed() const;
    edge_ocp_ptr_t compose() const;
};

class graph_model {
  public:
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

    model_node_ptr_t create_node(const node_ocp_ptr_t &base_prob = node_ocp::create(),
                                 const var_inarg_list &x = {},
                                 const var_inarg_list &u = {}) {
        auto node = model_node::create(base_prob);
        static_cast<void>(x);
        static_cast<void>(u);
        state_->dirty = true;
        node->owner_ = state_;
        node->id_ = state_->nodes.size();
        state_->nodes.emplace_back(node);
        return node;
    }

    model_node_ptr_t add_node(const node_ocp_ptr_t &base_prob = node_ocp::create(),
                              const var_inarg_list &x = {},
                              const var_inarg_list &u = {}) {
        return create_node(base_prob, x, u);
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

    std::vector<model_edge_ptr_t> add_path(const model_node_ptr_t &st,
                                           const model_node_ptr_t &ed,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        validate_node(ed);
        if (n_edges == 0) {
            throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
        }
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
                                           const node_ocp_ptr_t &ed_prob,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path(st, create_node(ed_prob), n_edges, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(const node_ocp_ptr_t &st_prob,
                                           const model_node_ptr_t &ed,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path(create_node(st_prob), ed, n_edges, base_prob);
    }

    std::vector<model_edge_ptr_t> add_path(const node_ocp_ptr_t &st_prob,
                                           const node_ocp_ptr_t &ed_prob,
                                           size_t n_edges,
                                           const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        return add_path(create_node(st_prob), create_node(ed_prob), n_edges, base_prob);
    }

    const model_node_ptr_t &node(size_t id) const { return state_->nodes.at(id); }
    const model_edge_ptr_t &edge(size_t id) const { return state_->edges.at(id); }

    node_ocp_ptr_t compose_terminal(const model_node_ptr_t &node_h) const {
        validate_node(node_h);
        auto composed = node_ocp::compose(node_h);
        composed->wait_until_ready();
        return composed;
    }

    edge_ocp_ptr_t compose(const model_edge_ptr_t &edge_h) const {
        validate_edge(edge_h);
        auto composed = edge_h->compose();
        composed->wait_until_ready();
        return composed;
    }

    std::vector<edge_ocp_ptr_t> compose_all() const {
        std::vector<edge_ocp_ptr_t> out;
        out.reserve(state_->edges.size());
        for (const auto &edge_h : state_->edges) {
            validate_edge(edge_h);
            auto composed = edge_h->compose();
            if (!state_->has_outgoing_edge(edge_h->ed_id_, edge_h->id_)) {
                for (const shared_expr &expr : edge_h->ed_node_prob()->exprs(__cost)) {
                    const auto *func = dynamic_cast<const generic_func *>(expr.get());
                    if (func == nullptr) {
                        continue;
                    }
                    const auto *cost_expr = dynamic_cast<const generic_cost *>(expr.get());
                    if (cost_expr != nullptr && cost_expr->terminal_add()) {
                        continue;
                    }
                    bool pure_state_cost = true;
                    for (const sym &arg : func->in_args()) {
                        if (arg.field() != __x) {
                            pure_state_cost = false;
                            break;
                        }
                    }
                    if (!pure_state_cost) {
                        continue;
                    }
                    auto lowered = expr.clone();
                    auto *lowered_func = dynamic_cast<generic_func *>(lowered.get());
                    if (lowered_func == nullptr) {
                        continue;
                    }
                    for (const sym &arg : lowered_func->in_args()) {
                        if (compose_trace_enabled()) {
                            fmt::print("materializing sink-node cost {} in composed ocp uid {}: {} -> {} (x_terminal -> incoming y)\n",
                                       expr->name(), composed->uid(), arg.name(), arg.next()->name());
                        }
                        lowered_func->substitute_argument(arg, arg.next());
                    }
                    composed->add(lowered);
                }
            }
            // Materialize/finalize sequentially so repeated lowered clones with the same
            // generated function name do not race in asynchronous codegen.
            composed->wait_until_ready();
            out.emplace_back(std::move(composed));
        }
        return out;
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

inline node_ocp_ptr_t model_node::compose() const {
    return node_ocp::compose(clone_node());
}

inline node_ocp_ptr_t model_node::compose_terminal() const {
    return compose();
}

inline model_node_ptr_t model_edge::st() const {
    return std::dynamic_pointer_cast<model_node>(st_node_prob());
}

inline model_node_ptr_t model_edge::ed() const {
    return std::dynamic_pointer_cast<model_node>(ed_node_prob());
}

inline edge_ocp_ptr_t model_edge::compose() const {
    const auto owner = owner_.lock();
    if (!owner) {
        throw std::runtime_error("graph_model edge is detached from its owner");
    }
    const bool st_has_previous_edge = owner->has_incoming_edge(st_id_, id_);
    return edge_ocp::compose(st_node_prob(), clone_edge(), ed_node_prob(), st_has_previous_edge);
}

} // namespace moto::model

#endif
