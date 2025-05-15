#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <atri/solver/ns_node.hpp>

namespace atri {
namespace ns_riccati {

void pre_solving_steps_0(node *cur);
void pre_solving_steps_1(node *cur);
void pre_solving_steps_2(node *prev, node *cur);
void backward_pass(node *cur, node *prev);
void post_solving_steps(node *cur);
void forward_rollout(node *cur, node *next);
void post_rollout_steps(node *cur);
} // namespace ns_riccati

} // namespace atri

#endif