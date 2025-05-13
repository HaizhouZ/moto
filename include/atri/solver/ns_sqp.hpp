#ifndef __NS_SQP__
#define __NS_SQP__

#include <atri/core/directed_graph.hpp>
#include <atri/ocp/core/shooting_node.hpp>

namespace atri {

struct ns_sqp {

    struct node : public shooting_node {
        node(problem_ptr_t prob);
    };

    void update();

    directed_graph<node> graph_;
    std::array<double, 6> timings{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

} // namespace atri

#endif