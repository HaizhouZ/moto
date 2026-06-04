#ifndef MOTO_MODEL_GRAPH_MODEL_HPP
#define MOTO_MODEL_GRAPH_MODEL_HPP

#include <memory>
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

    edge_ocp_ptr_t compose_interval(const edge_ocp_ptr_t &edge_h,
                                    bool include_terminal_sink_terms) const;

    std::vector<edge_ocp_ptr_t> edges_;
    size_t revision_ = 1;
};

} // namespace moto

#endif
