#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <atri/ocp/core/shooting_node.hpp>
#include <atri/solver/ns_riccati_data.hpp>
#include <list>

namespace atri {
namespace ns_riccati_solver {
inline auto &get_data(shooting_node *node) {
    return *static_cast<nullspace_riccati_data *>(node->data_.get());
}
void pre_solving_steps_1(shooting_node *cur);
void pre_solving_steps_2(shooting_node *cur, shooting_node *prev);
void backward_pass(shooting_node *cur, shooting_node *prev);
void post_solving_steps(shooting_node *cur);
void forward_rollout(shooting_node *cur, shooting_node *next);
void post_rollout_steps(shooting_node *cur);
} // namespace ns_riccati_solver

} // namespace atri

#endif