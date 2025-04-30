#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_sqp::update() {
    pre_solving_steps();
    backward_pass();
    post_solving_steps();
    forward_rollout();
    post_rollout_steps();
}
} // namespace atri
