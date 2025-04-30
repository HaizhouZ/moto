#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_sqp::update() {
    pre_solving_steps();
    backward_pass();
    post_solving_steps();
    forward_rollout();
    post_rollout_steps();
}
template <typename data_type>
nullspace_riccati_solver create(){
    // static_assert(std::is_base_of_v(data_type, nullspace_riccati_data));
};
} // namespace atri
