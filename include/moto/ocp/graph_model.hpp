#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <moto/ocp/problem.hpp>

namespace moto {

struct ns_sqp;
class graph_composer;

class graph_model {
  public:
    graph_model();

    node_view start_node() const;
    node_view st() const { return start_node(); }
    node_view ed() const;
    std::vector<stage_ocp_ptr_t> &stages() { return stages_; }
    const std::vector<stage_ocp_ptr_t> &stages() const { return stages_; }

  private:
    friend struct ns_sqp;
    friend class graph_composer;

    struct revision_state {
        std::atomic<size_t> revision = 1;
    };

    struct interval_record {
        stage_ocp_ptr_t stage;
        node_view start_boundary_view;
        std::vector<node_view> end_boundary_views;
    };

    struct topology_snapshot {
        size_t revision;
        std::shared_ptr<const std::vector<interval_record>> intervals;
    };

    topology_snapshot snapshot() const;
    size_t revision() const;
    void synchronize_topology_locked() const;
    void add_end_boundary_view(interval_record &record, const node_view &node) const;
    void invalidate() const;
    void attach_graph_callback(const stage_ocp_ptr_t &stage) const;
    bool same_node(const node_view &lhs, const node_view &rhs) const;

    std::shared_ptr<revision_state> revision_state_ = std::make_shared<revision_state>();
    stage_ocp_ptr_t start_stage_ = stage_ocp::create();
    stage_ocp_ptr_t end_stage_ = stage_ocp::create();
    std::vector<stage_ocp_ptr_t> stages_;
    mutable std::vector<interval_record> intervals_;
    mutable std::shared_ptr<const std::vector<interval_record>> topology_cache_;
    mutable std::mutex graph_state_mutex_;
};

} // namespace moto

#endif
