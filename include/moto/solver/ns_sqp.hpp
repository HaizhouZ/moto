#ifndef __NS_SQP__
#define __NS_SQP__

#include <moto/ocp/shooting_node.hpp>
#include <moto/solver/ns_riccati_data.hpp>

namespace moto {

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

} // namespace moto

#endif