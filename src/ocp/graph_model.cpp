#include <moto/ocp/graph_model.hpp>

#include <moto/ocp/impl/func.hpp>

#include <iterator>

namespace moto {

namespace {
void deactivate_inactive_primal_args_from_source(const stage_ocp_ptr_t &source,
                                                 const ocp_ptr_t &target) {
    if (!source) {
        return;
    }
    ocp::active_status_config config;
    for (field_t f : primal_fields) {
        for (const shared_expr &expr : target->exprs(f)) {
            if (source->contains(*expr) && !source->is_active(*expr)) {
                config.deactivate_list.emplace_back(*expr);
            }
        }
    }
    if (!config.empty()) {
        target->update_active_status(config);
    }
}
} // namespace

void graph_model::append_role_terms(const stage_ocp_ptr_t &source,
                                    stage_expr_role role,
                                    const ocp_ptr_t &target,
                                    term_placement placement) const {
    if (!source) {
        return;
    }
    for (field_t f : func_fields) {
        for (const shared_expr &expr : source->exprs(f)) {
            if (!source->has_role(*expr, role)) {
                continue;
            }
            if (placement == term_placement::direct) {
                target->add(expr);
            } else {
                target->add(expr.as<generic_func>().lower_expr_x_to_y_cached(
                    fmt::format("stage endpoint term {} materialization", expr->name()),
                    target->uid()));
            }
        }
    }
}

void graph_model::append_node_terms(const node_view &node,
                                    const ocp_ptr_t &target,
                                    term_placement placement) const {
    const auto source = node.stage();
    if (!source) {
        return;
    }
    source->wait_until_ready();
    append_role_terms(source, node.role(), target, placement);
}

void graph_model::add_end_boundary_view(interval_record &record, const node_view &node) const {
    if (node.expired()) {
        return;
    }
    for (const node_view &view : record.end_boundary_views) {
        if (same_node(view, node)) {
            return;
        }
    }
    record.end_boundary_views.push_back(node);
}

std::optional<size_t> graph_model::find_incoming_boundary_index(const node_view &node,
                                                                const char *missing_message) const {
    if (same_node(node, this->start_node())) {
        return std::nullopt;
    }
    for (size_t record_idx = 0; record_idx < intervals_.size(); ++record_idx) {
        const auto &record = intervals_[record_idx];
        for (const node_view &view : record.end_boundary_views) {
            if (same_node(view, node)) {
                return record_idx;
            }
        }
    }
    throw std::invalid_argument(missing_message);
}

graph_model::graph_model() {
    attach_graph_callback(start_stage_);
}

node_view graph_model::start_node() const {
    return start_stage_->st();
}

void graph_model::invalidate() {
    revision_state_->revision.fetch_add(1, std::memory_order_release);
}

void graph_model::attach_graph_callback(const stage_ocp_ptr_t &stage) {
    std::weak_ptr<revision_state> weak_revision = revision_state_;
    stage->set_mutation_callback([weak_revision] {
        if (auto revision = weak_revision.lock()) {
            revision->revision.fetch_add(1, std::memory_order_release);
        }
    });
}

bool graph_model::same_node(const node_view &lhs, const node_view &rhs) const {
    return lhs.role() == rhs.role() && lhs.stage().get() == rhs.stage().get();
}

std::vector<stage_ocp_ptr_t> graph_model::add_stage(const stage_ocp_ptr_t &stage,
                                                    size_t n_stages) {
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    const node_view start_node = tail_node_;
    validate_stage_chain_input(start_node, stage, n_stages);
    auto incoming_boundary_index = find_incoming_boundary_index(
        start_node,
        "graph_model tail node is not connected to an interval boundary");

    auto chain = build_stage_chain(start_node, stage, n_stages);
    return commit_stage_chain(std::move(chain), incoming_boundary_index, true);
}

std::vector<stage_ocp_ptr_t> graph_model::add_stages(const node_view &start_node,
                                                     const stage_ocp_ptr_t &stage,
                                                     size_t n_stages) {
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    validate_stage_chain_input(start_node, stage, n_stages);
    const bool advances_tail = same_node(start_node, tail_node_);
    auto incoming_boundary_index = find_incoming_boundary_index(
        start_node,
        "graph_model::add_stages start node must be sqp.start_node or an existing graph boundary");

    auto chain = build_stage_chain(start_node, stage, n_stages);
    return commit_stage_chain(std::move(chain), incoming_boundary_index, advances_tail);
}

void graph_model::validate_stage_chain_input(const node_view &start_node,
                                             const stage_ocp_ptr_t &stage,
                                             size_t n_stages) const {
    if (n_stages == 0) {
        throw std::invalid_argument("graph_model::add_stage expects n_stages >= 1");
    }
    if (!stage) {
        throw std::invalid_argument("graph_model::add_stage expects a non-null stage");
    }
    if (start_node.expired()) {
        throw std::invalid_argument("graph_model::add_stage expects a live start node");
    }
    if (intervals_.empty()) {
        if (!same_node(start_node, this->start_node())) {
            throw std::invalid_argument(
                "graph_model::add_stages first path must start from sqp.start_node");
        }
    }
}

graph_model::stage_chain graph_model::build_stage_chain(const node_view &start_node,
                                                        const stage_ocp_ptr_t &stage,
                                                        size_t n_stages) {
    stage_chain chain;
    chain.records.reserve(n_stages);
    chain.stages.reserve(n_stages);

    for (size_t i = 0; i < n_stages; ++i) {
        auto cloned = stage->clone();
        attach_graph_callback(cloned);
        if (!chain.records.empty()) {
            add_end_boundary_view(chain.records.back(), cloned->st());
        }
        interval_record record;
        record.stage = cloned;
        if (i == 0 && same_node(start_node, this->start_node())) {
            record.start_boundary_view = start_node;
        }
        add_end_boundary_view(record, cloned->ed());
        chain.records.push_back(record);
        chain.stages.push_back(cloned);
    }
    chain.tail = chain.records.back().stage->ed();
    return chain;
}

std::vector<stage_ocp_ptr_t> graph_model::commit_stage_chain(stage_chain &&chain,
                                                             std::optional<size_t> incoming_boundary_index,
                                                             bool advances_tail) {
    const node_view first_stage_start = chain.records.front().stage->st();
    if (const size_t capacity = intervals_.size() + chain.records.size(); capacity > intervals_.capacity()) {
        intervals_.reserve(capacity);
    }
    intervals_.insert(intervals_.end(),
                      std::make_move_iterator(chain.records.begin()),
                      std::make_move_iterator(chain.records.end()));
    if (incoming_boundary_index.has_value()) {
        add_end_boundary_view(intervals_[*incoming_boundary_index], first_stage_start);
    }
    if (advances_tail) {
        tail_node_ = chain.tail;
    }
    auto stages = std::move(chain.stages);
    interval_cache_.reset();
    invalidate();
    return stages;
}

graph_model::interval_snapshot graph_model::composed_intervals() const {
    for (;;) {
        size_t captured_revision = 0;
        std::vector<interval_record> interval_snapshot;
        {
            std::lock_guard<std::mutex> lock(graph_state_mutex_);
            captured_revision = revision();
            if (interval_cache_revision_ == captured_revision && interval_cache_) {
                return {captured_revision, interval_cache_};
            }
            if (intervals_.empty()) {
                throw std::runtime_error("graph_model expects a non-empty path");
            }
            interval_snapshot = intervals_;
        }

        auto intervals = std::make_shared<std::vector<ocp_ptr_t>>();
        intervals->reserve(interval_snapshot.size());
        for (const auto &record : interval_snapshot) {
            intervals->emplace_back(compose_stage(record));
        }

        std::lock_guard<std::mutex> lock(graph_state_mutex_);
        const size_t current_revision = revision();
        if (interval_cache_revision_ == current_revision && interval_cache_) {
            return {current_revision, interval_cache_};
        }
        if (current_revision == captured_revision) {
            interval_cache_ = intervals;
            interval_cache_revision_ = captured_revision;
            return {captured_revision, interval_cache_};
        }
    }
}

ocp_ptr_t graph_model::compose_stage(const interval_record &record) const {
    if (!record.stage) {
        throw std::runtime_error("graph_model found null stage");
    }
    record.stage->wait_until_ready();

    auto composed = ocp::create();
    composed->set_allow_inconsistent_dynamics(record.stage->allow_inconsistent_dynamics());
    composed->set_automatic_reorder_primal(record.stage->automatic_reorder_primal());

    append_role_terms(record.stage, stage_expr_role::interval, composed, term_placement::direct);
    if (!record.start_boundary_view.expired()) {
        append_node_terms(record.start_boundary_view, composed, term_placement::direct);
    }

    deactivate_inactive_primal_args_from_source(record.stage, composed);
    if (!record.start_boundary_view.expired()) {
        deactivate_inactive_primal_args_from_source(record.start_boundary_view.stage(), composed);
    }

    for (const node_view &view : record.end_boundary_views) {
        append_node_terms(view, composed, term_placement::lower_x_to_y);
    }
    composed->wait_until_ready();
    return composed;
}

} // namespace moto
