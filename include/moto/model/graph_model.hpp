#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <vector>

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
    bool has_incoming_edge(size_t node_id, size_t exclude_edge_id = static_cast<size_t>(-1)) const;

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
    var_list x() const;
    var_list u() const;
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
    var_list x() const;
    var_list u() const;
    var_list y() const;
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

    model_node_ptr_t add_node(const node_ocp_ptr_t &base_prob = node_ocp::create(),
                              const var_inarg_list &x = {},
                              const var_inarg_list &u = {}) {
        auto node = model_node::create(base_prob);
        static_cast<void>(x);
        static_cast<void>(u);
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
        auto edge = model_edge::create(base_prob);
        edge->owner_ = state_;
        edge->id_ = state_->edges.size();
        edge->st_id_ = st->id();
        edge->ed_id_ = ed->id();
        edge->bind_nodes(st, ed);
        state_->edges.emplace_back(edge);
        return edge;
    }

    const model_node_ptr_t &node(size_t id) const { return state_->nodes.at(id); }
    const model_edge_ptr_t &edge(size_t id) const { return state_->edges.at(id); }

    node_ocp_ptr_t compose_terminal(const model_node_ptr_t &node_h) const {
        validate_node(node_h);
        return node_ocp::compose(node_h);
    }

    edge_ocp_ptr_t compose(const model_edge_ptr_t &edge_h) const {
        validate_edge(edge_h);
        return edge_h->compose();
    }

    size_t num_nodes() const noexcept { return state_->nodes.size(); }
    size_t num_edges() const noexcept { return state_->edges.size(); }

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

inline var_list model_node::x() const {
    return graph_model::vars_from_prob(clone_node(), __x);
}

inline var_list model_node::u() const {
    return graph_model::vars_from_prob(clone_node(), __u);
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

inline var_list model_edge::x() const {
    return graph_model::vars_from_prob(clone_edge(), __x);
}

inline var_list model_edge::u() const {
    return graph_model::vars_from_prob(clone_edge(), __u);
}

inline var_list model_edge::y() const {
    return graph_model::vars_from_prob(clone_edge(), __y);
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
