#include <moto/ocp/graph_model.hpp>

#include <algorithm>
#include <unordered_map>

namespace moto {

namespace {
bool is_terminal_node_term(const shared_expr &expr) {
    if (!expr) {
        return false;
    }
    if (expr->field() == __cost) {
        return expr.as<generic_cost>().terminal_add();
    }
    if (in_field(expr->field(), constr_fields)) {
        return expr.as<generic_constr>().terminal_add();
    }
    return false;
}

bool is_pure_state_node_term(const shared_expr &expr, bool include_terminal = false) {
    if (!expr || expr->field() == __dyn || !in_field(expr->field(), func_fields)) {
        return false;
    }
    if (!include_terminal && is_terminal_node_term(expr)) {
        return false;
    }
    const auto &func = expr.as<generic_func>();
    bool has_x = false;
    for (const sym &arg : func.in_args()) {
        if (arg.field() == __u || arg.field() == __y) {
            return false;
        }
        if (arg.field() == __x) {
            has_x = true;
        }
    }
    return has_x;
}

} // namespace

struct graph_model::impl {
    std::vector<model_node_ptr_t> nodes;
    std::vector<model_edge_ptr_t> edges;
    size_t topology_revision = 1;

    void reserve(size_t node_capacity, size_t edge_capacity) {
        if (node_capacity > nodes.capacity()) nodes.reserve(node_capacity);
        if (edge_capacity > edges.capacity()) edges.reserve(edge_capacity);
    }

    void mark_topology_changed() noexcept { ++topology_revision; }

    size_t revision() const noexcept {
        size_t seed = topology_revision;
        const auto mix = [](size_t &s, size_t v) {
            s ^= v + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2);
        };
        for (const auto &node : nodes) {
            if (!node) continue;
            mix(seed, node->uid());
            mix(seed, node->formulation_version());
        }
        for (const auto &edge : edges) {
            if (!edge) continue;
            mix(seed, edge->uid());
            mix(seed, edge->formulation_version());
        }
        return seed;
    }

    std::unordered_map<size_t, size_t> node_ids_by_uid() const {
        std::unordered_map<size_t, size_t> out;
        out.reserve(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (!nodes[i]) {
                throw std::runtime_error("graph_model contains a null node handle");
            }
            out.emplace(nodes[i]->uid(), i);
        }
        return out;
    }
};

graph_model::graph_model() : state_(std::make_shared<impl>()) {}

void graph_model::reserve(size_t node_capacity, size_t edge_capacity) {
    state_->reserve(node_capacity, edge_capacity);
}

model_node_ptr_t graph_model::create_node(const node_ocp_ptr_t &base_prob) {
    auto node = base_prob ? base_prob->clone_node() : node_ocp::create();
    state_->mark_topology_changed();
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
    state_->mark_topology_changed();
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
    auto start_node_prob = edge_h->st_node_prob();
    auto end_node_prob = edge_h->ed_node_prob();
    if (start_node_prob) {
        start_node_prob->wait_until_ready();
    }
    if (end_node_prob) {
        end_node_prob->wait_until_ready();
    }
    if (!opts.source_config.empty()) {
        start_node_prob = start_node_prob ? start_node_prob->clone_node(opts.source_config) : node_ocp_ptr_t{};
    }

    if (start_node_prob) {
        ocp::active_status_config config;
        for (size_t f = 0; f < field::num; ++f) {
            for (const shared_expr &expr : start_node_prob->exprs(f)) {
                if (is_pure_state_node_term(expr, true)) {
                    config.deactivate_list.emplace_back(*expr);
                }
            }
        }
        if (!config.empty()) {
            start_node_prob = start_node_prob->clone_node(config);
        }
    }

    auto composed = edge_ocp::compose(start_node_prob, edge_h, node_ocp_ptr_t{}, false);
    composed->bind_nodes(start_node_prob, end_node_prob);
    if (end_node_prob) {
        for (size_t f = 0; f < field::num; ++f) {
            const auto ff = static_cast<field_t>(f);
            if (ff == __dyn || !in_field(ff, func_fields)) continue;
            for (const shared_expr &expr : end_node_prob->exprs(f)) {
                const bool is_terminal = is_terminal_node_term(expr);
                if (is_terminal && !opts.include_terminal_sink_terms) continue;
                if (!is_pure_state_node_term(expr, opts.include_terminal_sink_terms)) continue;
                if (is_terminal && expr.as<generic_func>().has_u_arg()) {
                    fmt::print(stderr,
                               "warning: terminal node term {} depends on u and cannot be lowered onto the final edge; ignoring it\n",
                               expr->name());
                    continue;
                }
                composed->add(expr.as<generic_func>().lower_expr_x_to_y_cached(
                    fmt::format("sink-node term {} materialization", expr->name()),
                    composed->uid()));
            }
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
        auto composed = compose_interval(edge_h, {
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
    graph.reserve(num_edges, num_edges);

    const auto node_ids = state_->node_ids_by_uid();
    std::vector<std::vector<size_t>> incoming(num_nodes), outgoing(num_nodes);
    for (size_t eid = 0; eid < num_edges; ++eid) {
        const auto &edge = state_->edges[eid];
        const auto &st_prob = edge ? edge->st_node_prob() : node_ocp_ptr_t{};
        const auto &ed_prob = edge ? edge->ed_node_prob() : node_ocp_ptr_t{};
        if (!st_prob || !ed_prob) {
            throw std::runtime_error("graph_model::realize_into found edge without bound endpoints");
        }
        incoming.at(node_ids.at(ed_prob->uid())).push_back(eid);
        outgoing.at(node_ids.at(st_prob->uid())).push_back(eid);
    }

    size_t head_node = static_cast<size_t>(-1);
    size_t tail_node = static_cast<size_t>(-1);
    for (size_t nid = 0; nid < num_nodes; ++nid) {
        const auto &inc = incoming[nid];
        const auto &out = outgoing[nid];
        if (inc.empty() && out.empty()) continue;
        if (inc.empty()) {
            if (out.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique outgoing edge from the source model node");
            if (head_node != static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single source path");
            head_node = nid;
        } else if (out.empty()) {
            if (inc.size() != 1) throw std::runtime_error("graph_model::realize_into expects a unique incoming edge for a sink model node");
            if (tail_node != static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single sink path");
            tail_node = nid;
        } else if (inc.size() != 1 || out.size() != 1) {
            throw std::runtime_error("graph_model::realize_into expects a single connected chain");
        }
    }
    if (head_node == static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single source path");
    if (tail_node == static_cast<size_t>(-1)) throw std::runtime_error("graph_model::realize_into expects a single sink path");

    std::vector<size_t> realized(num_edges, static_cast<size_t>(-1));
    for (size_t eid = 0; eid < num_edges; ++eid) {
        const auto &edge = state_->edges[eid];
        const auto &ed_prob = edge ? edge->ed_node_prob() : node_ocp_ptr_t{};
        if (!ed_prob) {
            throw std::runtime_error("graph_model::realize_into found edge without bound end node");
        }
        const size_t ed_id = node_ids.at(ed_prob->uid());
        const bool sink_final = outgoing[ed_id].empty();
        auto stage_ocp = std::static_pointer_cast<ocp>(compose_interval(edge, {
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

    for (size_t nid = 0; nid < num_nodes; ++nid) {
        const auto &inc = incoming[nid];
        const auto &out = outgoing[nid];
        if (!inc.empty() && !out.empty()) {
            graph.connect(realized[inc.front()], realized[out.front()]);
        }
    }

    graph.set_head(realized[outgoing[head_node].front()]);
    graph.set_tail(realized[incoming[tail_node].front()]);
}

void graph_model::realize_into(storage_interface &graph) const {
    realize_into(graph, stage_builder_t{});
}

size_t graph_model::revision() const noexcept { return state_->revision(); }

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
        auto next = (i + 1 == n_edges) ? ed : create_node(prev);
        edges.emplace_back(connect(prev, next, base_prob));
        prev = next;
    }
    return edges;
}

} // namespace moto
