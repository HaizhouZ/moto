#ifndef __NS_SQP__
#define __NS_SQP__

#include <atri/ocp/shooting_node.hpp>
#include <atri/solver/ns_riccati_data.hpp>

namespace atri {

struct ns_sqp {
    struct node_type : public shooting_node<ns_riccati::riccati_data> {
        node_type(const ocp_ptr_t &prob)
            : shooting_node<ns_riccati::riccati_data>(prob) {}
        node_type(const node_type &rhs) = default;
        node_type(node_type &&rhs) = default;
    };
    void update(size_t n_iter);
    void forward();

    directed_graph<node_type> graph_;
};

} // namespace atri

#endif