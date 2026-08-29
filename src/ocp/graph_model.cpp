#include <moto/ocp/graph_model.hpp>

#include <algorithm>

namespace moto {

graph_model::graph_model()
    : topology_cache_(std::make_shared<const std::vector<stage_ref>>()),
      start_cache_{start_stage_, start_stage_->composition_identity()},
      end_cache_{end_stage_, end_stage_->composition_identity()} {}

node_view graph_model::st() const {
    return start_stage_->st();
}

node_view graph_model::ed() const {
    return end_stage_->ed();
}

void graph_model::synchronize_topology_locked() const {
    bool unchanged = start_cache_.identity == start_stage_->composition_identity() &&
                     end_cache_.identity == end_stage_->composition_identity() &&
                     topology_cache_ && topology_cache_->size() == stages_.size();
    for (size_t i = 0; unchanged && i < stages_.size(); ++i) {
        const auto &stage = stages_[i];
        const auto &cached = (*topology_cache_)[i];
        unchanged = stage && cached.stage == stage &&
                    cached.identity == stage->composition_identity();
    }
    if (unchanged) {
        return;
    }

    std::vector<const stage_ocp *> unique;
    unique.reserve(stages_.size());
    auto replacement = std::make_shared<std::vector<stage_ref>>();
    replacement->reserve(stages_.size());
    for (const auto &stage : stages_) {
        if (!stage) {
            throw std::invalid_argument("sqp.stages expects non-null stages");
        }
        unique.push_back(stage.get());
        replacement->push_back({stage, stage->composition_identity()});
    }
    std::sort(unique.begin(), unique.end(), std::less<const stage_ocp *>{});
    if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
        throw std::invalid_argument(
            "sqp.stages expects distinct stage objects; use stage.copy()");
    }

    start_cache_ = {start_stage_, start_stage_->composition_identity()};
    end_cache_ = {end_stage_, end_stage_->composition_identity()};
    topology_cache_ = std::move(replacement);
    ++revision_;
}

size_t graph_model::revision() const {
    std::lock_guard lock(graph_state_mutex_);
    synchronize_topology_locked();
    return revision_;
}

graph_model::topology_snapshot graph_model::snapshot() const {
    std::lock_guard lock(graph_state_mutex_);
    synchronize_topology_locked();
    if (topology_cache_->empty()) {
        throw std::runtime_error("graph_model expects a non-empty path");
    }
    return {revision_, topology_cache_, start_cache_, end_cache_};
}

} // namespace moto
