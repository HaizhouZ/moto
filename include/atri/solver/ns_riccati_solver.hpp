#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <atri/ocp/shooting_node.hpp>
#include <atri/solver/ns_riccati_data.hpp>
#include <list>

namespace atri {

class nullspace_riccati_solver {

  public:
    nullspace_riccati_solver()
        : mem_(data_mgr::get<nullspace_riccati_data>()) {}
    void set_horizon(size_t N) { nodes_.resize(N); }
    static auto &get_data(shooting_node_ptr_t node) {
        return *std::static_pointer_cast<nullspace_riccati_data>(node->data_);
    }

  protected:
    void pre_solving_steps();
    void backward_pass();
    void post_solving_steps();
    void forward_rollout();
    void post_rollout_steps();

    data_mgr &mem_;
    std::vector<shooting_node_ptr_t> nodes_;
};

} // namespace atri

#endif