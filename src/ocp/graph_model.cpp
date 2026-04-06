#include <moto/ocp/graph_model.hpp>

#include <algorithm>

namespace moto {

struct graph_model::impl {
    std::vector<model_node_ptr_t> nodes;
    std::vector<model_edge_ptr_t> edges;
    bool topology_changed = true;

    void reserve(size_t node_capacity, size_t edge_capacity) {
        if (node_capacity > nodes.capacity()) nodes.reserve(node_capacity);
        if (edge_capacity > edges.capacity()) edges.reserve(edge_capacity);
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

    size_t find_node_id_by_uid(size_t uid) const {
        const auto it = std::find_if(nodes.begin(), nodes.end(), [&](const model_node_ptr_t &node) {
            return node && node->uid() == uid;
        });
        if (it == nodes.end()) {
            throw std::runtime_error("graph_model cannot resolve node id from uid");
        }
        return static_cast<size_t>(std::distance(nodes.begin(), it));
    }

    bool has_incoming_edge(size_t node_id, size_t exclude_edge_id) const {
        const auto uid = nodes.at(node_id)->uid();
        for (size_t i = 0; i < edges.size(); ++i) {
            if (i == exclude_edge_id) continue;
            const auto &ed = edges[i] ? edges[i]->ed_node_prob() : node_ocp_ptr_t{};
            if (ed && ed->uid() == uid) return true;
        }
        return false;
    }

    bool has_outgoing_edge(size_t node_id, size_t exclude_edge_id) const {
        const auto uid = nodes.at(node_id)->uid();
        for (size_t i = 0; i < edges.size(); ++i) {
            if (i == exclude_edge_id) continue;
            const auto &st = edges[i] ? edges[i]->st_node_prob() : node_ocp_ptr_t{};
            if (st && st->uid() == uid) return true;
        }
        return false;
    }
};

graph_model::graph_model() : state_(std::make_shared<impl>()) {}

void graph_model::reserve(size_t node_capacity, size_t edge_capacity) {
    state_->reserve(node_capacity, edge_capacity);
}

model_node_ptr_t graph_model::create_node(const node_ocp_ptr_t &base_prob) {
    auto node = base_prob ? base_prob->clone_node() : node_ocp::create();
    state_->topology_changed = true;
    state_->nodes.emplace_back(node);
    return node;
}

model_edge_ptr_t graph_model::create_edge(const edge_ocp_ptr_t &base_prob) {
    auto st = create_node();
    auto ed = create_node();
    return connect(st, ed, base_prob);
}

model_edge_ptr_t graph_model::connect(const model_node_ptr_t &st,
                                      const model_node_ptr_t &ed,
                                      const edge_ocp_ptr_t &base_prob) {
    validate_node(st);
    validate_node(ed);
    state_->topology_changed = true;
    auto edge = base_prob ? base_prob->clone_edge() : edge_ocp::create();
    edge->bind_nodes(st, ed);
    state_->edges.emplace_back(edge);
    return edge;
}

std::vector<model_edge_ptr_t> graph_model::add_path(const model_node_ptr_t &st,
                                                     const model_node_ptr_t &ed,
                                                     size_t n_edges,
                                                     const edge_ocp_ptr_t &base_prob) {
    validate_node(st);
    validate_node(ed);
    return add_path_impl(st, ed, n_edges, base_prob);
}

std::vector<model_edge_ptr_t> graph_model::add_path(const model_node_ptr_t &st,
                                                     size_t n_edges,
                                                     const edge_ocp_ptr_t &base_prob) {
    validate_node(st);
    return add_path_impl(st, create_node(), n_edges, base_prob);
}

std::vector<model_edge_ptr_t> graph_model::add_path(size_t n_edges,
                                                     const edge_ocp_ptr_t &base_prob) {
    return add_path_impl(create_node(), create_node(), n_edges, base_prob);
}

const model_node_ptr_t &graph_model::node(size_t id) const { return state_->nodes.at(id); }
const model_edge_ptr_t &graph_model::edge(size_t id) const { return state_->edges.at(id); }

node_ocp_ptr_t graph_model::compose_terminal(const model_node_ptr_t &node_h) const {
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

edge_ocp_ptr_t graph_model::compose(const model_edge_ptr_t &edge_h) const {
    return compose_interval(edge_h, interval_compose_options{});
}

edge_ocp_ptr_t graph_model::compose_interval(const model_edge_ptr_t &edge_h,
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

std::vector<edge_ocp_ptr_t> graph_model::compose_all() const {
    std::vector<edge_ocp_ptr_t> out;
    out.reserve(state_->edges.size());
    for (size_t eid = 0; eid < state_->edges.size(); ++eid) {
        const auto &edge_h = state_->edges[eid];
        validate_edge(edge_h);
        const auto &ed_prob = edge_h->ed_node_prob();
        if (!ed_prob) {
            throw std::runtime_error("graph_model::compose_all found edge without bound end node");
        }
        const size_t ed_id = state_->find_node_id_by_uid(ed_prob->uid());
        const bool sink_final = !state_->has_outgoing_edge(ed_id, eid);
        auto composed = compose_interval(edge_h, {
            .materialize_sink_terms = sink_final,
            .include_terminal_sink_terms = false,
        });
        composed->wait_until_ready();
        out.emplace_back(std::move(composed));
    }
    return out;
}

void graph_model::realize_into(storage_interface &graph,
                               const stage_builder_t &stage_builder) const {
    graph.clear();
    const size_t num_nodes = state_->nodes.size();
    const size_t num_edges = state_->edges.size();

    std::vector<std::vector<size_t>> incoming(num_nodes), outgoing(num_nodes);
    for (size_t eid = 0; eid < num_edges; ++eid) {
        const auto &edge = state_->edges[eid];
        const auto &st_prob = edge ? edge->st_node_prob() : node_ocp_ptr_t{};
        const auto &ed_prob = edge ? edge->ed_node_prob() : node_ocp_ptr_t{};
        if (!st_prob || !ed_prob) {
            throw std::runtime_error("graph_model::realize_into found edge without bound endpoints");
        }
        incoming[state_->find_node_id_by_uid(ed_prob->uid())].push_back(eid);
        outgoing[state_->find_node_id_by_uid(st_prob->uid())].push_back(eid);
    }

    std::vector<size_t> realized(num_edges, static_cast<size_t>(-1));
    for (size_t eid = 0; eid < num_edges; ++eid) {
        const auto &edge = state_->edges[eid];
        const auto &ed_prob = edge ? edge->ed_node_prob() : node_ocp_ptr_t{};
        if (!ed_prob) {
            throw std::runtime_error("graph_model::realize_into found edge without bound end node");
        }
        const size_t ed_id = state_->find_node_id_by_uid(ed_prob->uid());
        const bool sink_final = outgoing[ed_id].empty();
        auto stage_ocp = std::static_pointer_cast<ocp>(compose_interval(edge, {
            .materialize_sink_terms = sink_final,
            .include_terminal_sink_terms = sink_final,
        }));
        if (stage_builder) {
            stage_ocp = stage_builder(stage_ocp);
        }
        if (!stage_ocp) {
            throw std::runtime_error("graph_model::realize_into stage_builder returned null stage_ocp");
        }
        realized[eid] = graph.add_stage(stage_ocp);
    }

    size_t head = static_cast<size_t>(-1);
    size_t tail = static_cast<size_t>(-1);
    for (size_t nid = 0; nid < num_nodes; ++nid) {
        const auto &inc = incoming[nid];
        const auto &out = outgoing[nid];
        if (inc.empty() && out.empty()) continue;
        if (inc.empty()) {
            if (out.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique outgoing edge from the source model node");
            if (head != static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single source path");
            head = realized[out.front()];
        } else if (out.empty()) {
            if (inc.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique incoming edge for a sink model node");
            if (tail != static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single sink path");
            tail = realized[inc.front()];
        } else {
            for (size_t in_eid : inc)
                for (size_t out_eid : out)
                    graph.connect(realized[in_eid], realized[out_eid]);
        }
    }

    if (head == static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single source path");
    if (tail == static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single sink path");
    graph.set_head(head);
    graph.set_tail(tail);
    state_->topology_changed = false;
}

void graph_model::realize_into(storage_interface &graph) const {
    realize_into(graph, stage_builder_t{});
}

bool graph_model::topology_changed() const noexcept { return state_->topology_changed; }

size_t graph_model::num_nodes() const noexcept { return state_->nodes.size(); }
size_t graph_model::num_edges() const noexcept { return state_->edges.size(); }

bool graph_model::is_terminal_expr(const shared_expr &expr, field_t f) {
    if (f == __cost) {
        return expr.as<generic_cost>().terminal_add();
    }
    if (in_field(f, constr_fields)) {
        return expr.as<generic_constr>().terminal_add();
    }
    return false;
}

void graph_model::validate_node(const model_node_ptr_t &node_h) const {
    if (!node_h) {
        throw std::runtime_error("invalid graph_model node handle");
    }
    const auto it = std::find(state_->nodes.begin(), state_->nodes.end(), node_h);
    if (it == state_->nodes.end()) {
        throw std::runtime_error("invalid graph_model node handle");
    }
}

void graph_model::validate_edge(const model_edge_ptr_t &edge_h) const {
    if (!edge_h) {
        throw std::runtime_error("invalid graph_model edge handle");
    }
    const auto it = std::find_if(state_->edges.begin(), state_->edges.end(), [&](const model_edge_ptr_t &edge) {
        return edge == edge_h;
    });
    if (it == state_->edges.end()) {
        throw std::runtime_error("invalid graph_model edge handle");
    }
}

std::vector<model_edge_ptr_t> graph_model::add_path_impl(const model_node_ptr_t &st,
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

void graph_model::materialize_sink_terms(const node_ocp &sink_node,
                                         edge_ocp &composed,
                                         bool include_terminal) {
    for (const shared_expr &expr : sink_node.exprs(__cost)) {
        if (!is_terminal_expr(expr, __cost)) {
            lower_expr_into(expr, composed);
        }
    }

    if (!include_terminal) return;
    for (size_t f = 0; f < field::num; ++f) {
        const auto ff = static_cast<field_t>(f);
        if (ff == __dyn || !in_field(ff, func_fields)) continue;
        for (const shared_expr &expr : sink_node.exprs(f)) {
            if (!is_terminal_expr(expr, ff)) continue;
            if (expr.as<generic_func>().has_u_arg()) {
                fmt::print(stderr, "warning: terminal node term {} depends on u and cannot be lowered onto the final edge; ignoring it\n",
                           expr->name());
                continue;
            }
            lower_expr_into(expr, composed);
        }
    }
}

void graph_model::lower_expr_into(const shared_expr &expr, edge_ocp &composed) {
    composed.add(expr.as<generic_func>().lower_expr_x_to_y_cached(
        fmt::format("sink-node term {} materialization", expr->name()),
        composed.uid()));
}

} // namespace moto
