#include <moto/ocp/graph_model.hpp>

#include <unordered_set>

namespace moto {

void graph_model::add_end_boundary_view(interval_record &record, const node_view &node) const {
    if (!node) {
        return;
    }
    for (const node_view &view : record.end_boundary_views) {
        if (same_node(view, node)) {
            return;
        }
    }
    record.end_boundary_views.push_back(node);
}

graph_model::graph_model() {
    attach_graph_callback(start_stage_);
    attach_graph_callback(end_stage_);
}

node_view graph_model::start_node() const {
    return start_stage_->st();
}

node_view graph_model::ed() const {
    return end_stage_->ed();
}

void graph_model::synchronize_topology_locked() const {
    bool unchanged = stages_.size() == intervals_.size();
    for (size_t i = 0; unchanged && i < stages_.size(); ++i) {
        unchanged = stages_[i] == intervals_[i].stage;
    }
    if (unchanged) {
        return;
    }

    std::unordered_set<const stage_ocp *> unique;
    for (const auto &stage : stages_) {
        if (!stage) {
            throw std::invalid_argument("sqp.stages expects non-null stages");
        }
        if (!unique.insert(stage.get()).second) {
            throw std::invalid_argument("sqp.stages expects distinct stage objects; use stage.copy()");
        }
    }

    std::vector<interval_record> replacement;
    replacement.reserve(stages_.size());
    for (size_t i = 0; i < stages_.size(); ++i) {
        attach_graph_callback(stages_[i]);
        if (!replacement.empty()) {
            add_end_boundary_view(replacement.back(), stages_[i]->st());
        }
        interval_record record;
        record.stage = stages_[i];
        if (i == 0) {
            record.start_boundary_view = start_node();
        }
        add_end_boundary_view(record, stages_[i]->ed());
        replacement.push_back(std::move(record));
    }
    if (!replacement.empty()) {
        add_end_boundary_view(replacement.back(), ed());
    }
    intervals_ = std::move(replacement);
    topology_cache_.reset();
    invalidate();
}

size_t graph_model::revision() const {
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    synchronize_topology_locked();
    return revision_state_->revision.load(std::memory_order_acquire);
}

void graph_model::invalidate() const {
    revision_state_->revision.fetch_add(1, std::memory_order_release);
}

void graph_model::attach_graph_callback(const stage_ocp_ptr_t &stage) const {
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

graph_model::topology_snapshot graph_model::snapshot() const {
    std::lock_guard<std::mutex> lock(graph_state_mutex_);
    synchronize_topology_locked();
    if (intervals_.empty()) {
        throw std::runtime_error("graph_model expects a non-empty path");
    }
    if (!topology_cache_) {
        topology_cache_ = std::make_shared<const std::vector<interval_record>>(intervals_);
    }
    return {revision_state_->revision.load(std::memory_order_acquire), topology_cache_};
}

} // namespace moto
