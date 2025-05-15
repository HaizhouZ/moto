#ifndef ATRI_SOLVER_NS_NODE_HPP
#define ATRI_SOLVER_NS_NODE_HPP

#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/ns_riccati_data.hpp>

namespace atri {
namespace ns_riccati_solver {

template <typename node_type>
inline auto &get_data(node_type *node) {
    return *static_cast<nullspace_riccati_data *>(node->data_.get());
}

struct node : public shooting_node {
    node(problem_ptr_t prob)
        : shooting_node(prob, data_mgr::get<nullspace_riccati_data>()) {}
};

} // namespace ns_riccati_solver
} // namespace atri

#endif // ATRI_SOLVER_NS_NODE_HPP