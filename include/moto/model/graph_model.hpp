#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <vector>

#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>

namespace moto::model {

class graph_model;
struct node_handle;
struct edge_handle;

struct model_node {
    node_ocp_ptr_t base_prob = node_ocp::create();
};

struct model_edge {
    size_t st = static_cast<size_t>(-1);
    size_t ed = static_cast<size_t>(-1);
    edge_ocp_ptr_t base_prob = edge_ocp::create();
};

struct graph_model_state {
    std::vector<model_node> nodes;
    std::vector<model_edge> edges;

    size_t find_node_id(const node_ocp_ptr_t &prob) const {
        if (!prob) {
            throw std::runtime_error("graph_model cannot resolve a null node problem");
        }
        const auto it = std::find_if(nodes.begin(), nodes.end(), [&](const model_node &node) {
            return node.base_prob == prob;
        });
        if (it == nodes.end()) {
            throw std::runtime_error("graph_model cannot resolve node handle from node problem");
        }
        return static_cast<size_t>(std::distance(nodes.begin(), it));
    }
};

struct node_handle {
    std::shared_ptr<graph_model_state> owner;
    size_t id = static_cast<size_t>(-1);

    explicit operator bool() const noexcept { return owner != nullptr && id != static_cast<size_t>(-1); }

    model_node &data() const;
    node_ocp_ptr_t base_prob() const;
    var_list x() const;
    var_list u() const;

    node_handle &add(const shared_expr &expr);
    node_handle &add_terminal(const shared_expr &expr);
    node_ocp_ptr_t compose() const;
    node_ocp_ptr_t compose_terminal() const;
};

struct edge_handle {
    std::shared_ptr<graph_model_state> owner;
    size_t id = static_cast<size_t>(-1);

    explicit operator bool() const noexcept { return owner != nullptr && id != static_cast<size_t>(-1); }

    model_edge &data() const;
    edge_ocp_ptr_t base_prob() const;
    node_handle st() const;
    node_handle ed() const;
    var_list x() const;
    var_list u() const;
    var_list y() const;

    edge_handle &add(const shared_expr &expr);
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

    node_handle add_node(const node_ocp_ptr_t &base_prob = node_ocp::create(),
                         const var_inarg_list &x = {},
                         const var_inarg_list &u = {}) {
        model_node node;
        node.base_prob = base_prob ? base_prob : node_ocp::create();
        static_cast<void>(x);
        static_cast<void>(u);
        state_->nodes.emplace_back(std::move(node));
        return node_handle{state_, state_->nodes.size() - 1};
    }

    edge_handle connect(node_handle st, node_handle ed, const edge_ocp_ptr_t &base_prob = edge_ocp::create()) {
        validate_node(st);
        validate_node(ed);
        model_edge edge;
        edge.st = st.id;
        edge.ed = ed.id;
        edge.base_prob = base_prob ? base_prob : edge_ocp::create();
        edge.base_prob->bind_nodes(st.base_prob(), ed.base_prob());
        state_->edges.emplace_back(std::move(edge));
        return edge_handle{state_, state_->edges.size() - 1};
    }

    model_node &node(size_t id) { return state_->nodes.at(id); }
    const model_node &node(size_t id) const { return state_->nodes.at(id); }
    model_edge &edge(size_t id) { return state_->edges.at(id); }
    const model_edge &edge(size_t id) const { return state_->edges.at(id); }

    node_ocp_ptr_t compose_terminal(node_handle node_h) const {
        validate_node(node_h);
        const auto &node = state_->nodes.at(node_h.id);
        return node_ocp::compose(node.base_prob);
    }

    edge_ocp_ptr_t compose(edge_handle edge_h) const {
        validate_edge(edge_h);
        const auto &edge = state_->edges.at(edge_h.id);
        return edge.base_prob->compose();
    }

    size_t num_nodes() const noexcept { return state_->nodes.size(); }
    size_t num_edges() const noexcept { return state_->edges.size(); }

  private:
    friend struct node_handle;
    friend struct edge_handle;

    void validate_node(node_handle node_h) const {
        if (!node_h || node_h.owner.get() != state_.get() || node_h.id >= state_->nodes.size()) {
            throw std::runtime_error("invalid graph_model node handle");
        }
    }

    void validate_edge(edge_handle edge_h) const {
        if (!edge_h || edge_h.owner.get() != state_.get() || edge_h.id >= state_->edges.size()) {
            throw std::runtime_error("invalid graph_model edge handle");
        }
    }

    std::shared_ptr<graph_model_state> state_;
};

inline model_node &node_handle::data() const {
    if (!owner || id >= owner->nodes.size()) {
        throw std::runtime_error("invalid graph_model node handle");
    }
    return owner->nodes.at(id);
}

inline node_ocp_ptr_t node_handle::base_prob() const {
    return data().base_prob;
}

inline var_list node_handle::x() const {
    return graph_model::vars_from_prob(base_prob(), __x);
}

inline var_list node_handle::u() const {
    return graph_model::vars_from_prob(base_prob(), __u);
}

inline node_handle &node_handle::add(const shared_expr &expr) {
    base_prob()->add(expr);
    return *this;
}

inline node_handle &node_handle::add_terminal(const shared_expr &expr) {
    base_prob()->add_terminal(expr);
    return *this;
}

inline node_ocp_ptr_t node_handle::compose() const {
    if (!owner || id >= owner->nodes.size()) {
        throw std::runtime_error("invalid graph_model node handle");
    }
    return node_ocp::compose(base_prob());
}

inline node_ocp_ptr_t node_handle::compose_terminal() const {
    if (!owner || id >= owner->nodes.size()) {
        throw std::runtime_error("invalid graph_model node handle");
    }
    graph_model g(owner);
    return g.compose_terminal(*this);
}

inline model_edge &edge_handle::data() const {
    if (!owner || id >= owner->edges.size()) {
        throw std::runtime_error("invalid graph_model edge handle");
    }
    return owner->edges.at(id);
}

inline edge_ocp_ptr_t edge_handle::base_prob() const {
    return data().base_prob;
}

inline node_handle edge_handle::st() const {
    return node_handle{owner, owner->find_node_id(base_prob()->st_node_prob())};
}

inline node_handle edge_handle::ed() const {
    return node_handle{owner, owner->find_node_id(base_prob()->ed_node_prob())};
}

inline var_list edge_handle::x() const {
    return graph_model::vars_from_prob(base_prob(), __x);
}

inline var_list edge_handle::u() const {
    return graph_model::vars_from_prob(base_prob(), __u);
}

inline var_list edge_handle::y() const {
    return graph_model::vars_from_prob(base_prob(), __y);
}

inline edge_handle &edge_handle::add(const shared_expr &expr) {
    base_prob()->add(expr);
    return *this;
}

inline edge_ocp_ptr_t edge_handle::compose() const {
    if (!owner || id >= owner->edges.size()) {
        throw std::runtime_error("invalid graph_model edge handle");
    }
    graph_model g(owner);
    return g.compose(*this);
}

} // namespace moto::model

#endif
