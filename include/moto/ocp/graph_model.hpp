#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

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

    node_view st() const;
    node_view ed() const;
    std::vector<stage_ocp_ptr_t> &stages() { return stages_; }
    const std::vector<stage_ocp_ptr_t> &stages() const { return stages_; }

  private:
    friend struct ns_sqp;
    friend class graph_composer;

    struct stage_ref {
        stage_ocp_ptr_t stage;
        stage_composition_identity_ptr_t identity;
    };

    struct topology_snapshot {
        size_t revision;
        std::shared_ptr<const std::vector<stage_ref>> stages;
        stage_ref start;
        stage_ref end;
    };

    topology_snapshot snapshot() const;
    size_t revision() const;
    void synchronize_topology_locked() const;
    stage_ocp_ptr_t start_stage_ = stage_ocp::create();
    stage_ocp_ptr_t end_stage_ = stage_ocp::create();
    std::vector<stage_ocp_ptr_t> stages_;
    mutable std::shared_ptr<const std::vector<stage_ref>> topology_cache_;
    mutable stage_ref start_cache_, end_cache_;
    mutable size_t revision_ = 1;
    mutable std::mutex graph_state_mutex_;
};

} // namespace moto

#endif
