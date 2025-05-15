#ifndef __NS_SQP__
#define __NS_SQP__

#include <atri/core/directed_graph.hpp>
#include <atri/solver/ns_node.hpp>

namespace atri {

struct ns_sqp {
    using node_type = atri::ns_riccati_solver::node;

    void update(size_t n_iter);

    directed_graph<node_type> graph_;
    // std::array<double, 7> timings{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

} // namespace atri

#endif