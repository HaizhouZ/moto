#ifndef MOTO_SOLVER_SOFT_CONSTR_HPP
#define MOTO_SOLVER_SOFT_CONSTR_HPP

#include <moto/ocp/constr.hpp>

namespace moto {
struct soft_constr_data : public constr_data {
    std::vector<vector_ref> prim_step_; // to be set
    soft_constr_data(constr_data &&d)
        : constr_data(std::move(d)) {
    }
};

class soft_constr_impl : public constr_impl {
  public:
    using constr_impl::constr_impl;
    virtual void initialize(soft_constr_data &data) {}
    virtual void post_rollout(soft_constr_data &data) {};
};

} // namespace moto

#endif // MOTO_SOLVER_SOFT_CONSTR_HPP