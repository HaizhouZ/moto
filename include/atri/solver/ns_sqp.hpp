#ifndef __NS_SQP__
#define __NS_SQP__

#include <atri/solver/ns_riccati_solver.hpp>

namespace atri {

struct nullspace_sqp : public nullspace_riccati_solver {
    void update();
};

} // namespace atri

#endif