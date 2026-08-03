#include <moto/ocp/graph_model.hpp>

#include <iterator>

namespace moto {

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

std::vector<std::vector<stage_ocp_ptr_t>>
graph_model::add_phases(const std::vector<phase> &phases) {
    if (phases.empty()) {
        throw std::invalid_argument("graph_model::add_phases expects at least one phase");
    }
    for (const auto &[stage, count] : phases) {
        if (!stage || count == 0) {
            throw std::invalid_argument("graph_model::add_phases expects non-null stages and counts >= 1");
        }
    }
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    const node_view start_node = tail_node_;
    const auto incoming_boundary_index = find_incoming_boundary_index(
        start_node, "graph_model tail node is not connected to an interval boundary");

    stage_chain combined;
    std::vector<std::vector<stage_ocp_ptr_t>> result;
    result.reserve(phases.size());
    node_view cursor = start_node;
    for (const auto &[stage, count] : phases) {
        auto chain = build_stage_chain(cursor, stage, count);
        if (!combined.records.empty()) {
            add_end_boundary_view(combined.records.back(), chain.records.front().stage->st());
        }
        result.push_back(chain.stages);
        combined.records.insert(combined.records.end(),
                                std::make_move_iterator(chain.records.begin()),
                                std::make_move_iterator(chain.records.end()));
        combined.stages.insert(combined.stages.end(), chain.stages.begin(), chain.stages.end());
        combined.tail = chain.tail;
        cursor = chain.tail;
    }
    commit_stage_chain(std::move(combined), incoming_boundary_index, true);
    return result;
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
    topology_cache_.reset();
    invalidate();
    return stages;
}

graph_model::topology_snapshot graph_model::snapshot() const {
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    if (intervals_.empty()) {
        throw std::runtime_error("graph_model expects a non-empty path");
    }
    if (!topology_cache_) {
        topology_cache_ = std::make_shared<const std::vector<interval_record>>(intervals_);
    }
    return {revision(), topology_cache_};
}

} // namespace moto
