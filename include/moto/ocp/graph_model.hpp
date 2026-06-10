#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <moto/ocp/problem.hpp>

namespace moto {

struct ns_sqp;

class graph_model {
  public:
    graph_model() = default;

    void add_path(const node_ocp_ptr_t &stage_prob,
                  const node_ocp_ptr_t &next_prob,
                  const edge_ocp_ptr_t &edge_prob,
                  size_t n_edges);

  private:
    friend struct ns_sqp;

    struct interval_snapshot {
        size_t revision;
        std::shared_ptr<const std::vector<ocp_ptr_t>> intervals;
    };

    interval_snapshot composed_intervals() const;
    size_t revision() const noexcept { return revision_.load(std::memory_order_acquire); }
    edge_ocp_ptr_t compose_interval(const edge_ocp_ptr_t &edge,
                                    bool include_terminal_sink_terms) const;

    std::vector<edge_ocp_ptr_t> edges_;
    std::atomic<size_t> revision_ = 1;
    mutable std::shared_ptr<const std::vector<ocp_ptr_t>> interval_cache_;
    mutable size_t interval_cache_revision_ = 0;
    mutable std::mutex graph_state_mutex_;
};

} // namespace moto

#endif
