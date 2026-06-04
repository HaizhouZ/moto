#include <moto/ocp/graph_model.hpp>

#include <moto/ocp/constr.hpp>
#include <moto/ocp/cost.hpp>
#include <moto/ocp/impl/func.hpp>

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

void append_node_terms(const node_ocp_ptr_t &node_prob,
                       const edge_ocp_ptr_t &edge_prob) {
    for (field_t f : func_fields) {
        for (const shared_expr &expr : node_prob->exprs(f)) {
            edge_prob->add(expr);
        }
    }
}

} // namespace

void graph_model::add_path(const node_ocp_ptr_t &stage_prob,
                           const node_ocp_ptr_t &next_prob,
                           const edge_ocp_ptr_t &edge_prob,
                           size_t n_edges) {
    if (n_edges == 0) {
        throw std::invalid_argument("graph_model::add_path expects n_edges >= 1");
    }
    if (!stage_prob || !next_prob || !edge_prob) {
        throw std::invalid_argument("graph_model::add_path expects non-null start node, end node, and edge stages");
    }
    if (const size_t edge_capacity = edges_.size() + n_edges; edge_capacity > edges_.capacity()) {
        edges_.reserve(edge_capacity);
    }
    ++revision_;

    auto prev = stage_prob->clone_node();
    auto final_node = next_prob->clone_node();
    for (size_t i = 0; i < n_edges; ++i) {
        auto next = (i + 1 == n_edges) ? final_node : stage_prob->clone_node();
        auto edge = edge_prob->clone_edge();
        edge->bind_nodes(prev, next);
        edges_.emplace_back(edge);
        prev = next;
    }
}

edge_ocp_ptr_t graph_model::compose_interval(const edge_ocp_ptr_t &edge_h,
                                             bool include_terminal_sink_terms) const {
    edge_h->wait_until_ready();
    auto start_node_prob = edge_h->st_node_prob();
    auto end_node_prob = edge_h->ed_node_prob();
    if (!start_node_prob || !end_node_prob) {
        throw std::runtime_error("graph_model::compose_interval found edge without bound endpoints");
    }
    start_node_prob->wait_until_ready();
    end_node_prob->wait_until_ready();

    ocp::active_status_config config;
    for (field_t f : func_fields) {
        for (const shared_expr &expr : start_node_prob->exprs(f)) {
            if (is_pure_state_node_term(expr, true)) {
                config.deactivate_list.emplace_back(*expr);
            }
        }
    }
    if (!config.empty()) {
        start_node_prob = start_node_prob->clone_node(config);
    }

    auto composed = edge_h->clone_edge();
    config = {};
    for (size_t f = 0; f < field::num_prim; ++f) {
        for (const shared_expr &expr : edge_h->exprs(f)) {
            if (start_node_prob->contains(*expr) &&
                !start_node_prob->is_active(*expr)) {
                config.deactivate_list.emplace_back(*expr);
            }
        }
    }
    if (!config.empty()) {
        composed->update_active_status(config);
    }
    append_node_terms(start_node_prob, composed);
    composed->bind_nodes(start_node_prob, end_node_prob);
    for (field_t f : func_fields) {
        if (f == __dyn) continue;
        for (const shared_expr &expr : end_node_prob->exprs(f)) {
            const bool is_terminal = is_terminal_node_term(expr);
            if (is_terminal && !include_terminal_sink_terms) continue;
            if (!is_pure_state_node_term(expr, include_terminal_sink_terms)) continue;
            composed->add(expr.as<generic_func>().lower_expr_x_to_y_cached(
                fmt::format("sink-node term {} materialization", expr->name()),
                composed->uid()));
        }
    }
    composed->wait_until_ready();
    return composed;
}

} // namespace moto
