#ifndef ATRI_SOLVER_FWD_HPP
#define ATRI_SOLVER_FWD_HPP
#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/ns_riccati_data.hpp>

namespace atri {
namespace ns_riccati_solver {

inline auto &get_data(shooting_node *node) {
    return *static_cast<nullspace_riccati_data *>(node->data_.get());
}
} // namespace ns_riccati_solver
} // namespace atri

#endif // ATRI_SOLVER_FWD_HPP