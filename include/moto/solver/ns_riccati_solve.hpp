#ifndef __NS_RICCATI_SOLVER__
#define __NS_RICCATI_SOLVER__

#include <moto/solver/ns_riccati_data.hpp>

namespace moto {
namespace nullsp_kkt_solve {

void pre_solving_steps_0(riccati_data *cur);
void pre_solving_steps_1(riccati_data *cur);
void pre_solving_steps_2(riccati_data *prev, riccati_data *cur);
void backward_pass(riccati_data *cur, riccati_data *prev);
void post_solving_steps(riccati_data *cur);
void forward_rollout(riccati_data *cur, riccati_data *next);
void post_rollout_steps(riccati_data *cur);
} // namespace nullsp_kkt_solve

} // namespace moto

#endif