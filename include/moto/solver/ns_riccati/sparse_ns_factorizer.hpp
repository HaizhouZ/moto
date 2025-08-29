#ifndef MOTO_SOLVER_NS_RICCATI_SPARSE_NS_FACTORIZER_HPP
#define MOTO_SOLVER_NS_RICCATI_SPARSE_NS_FACTORIZER_HPP

#include <moto/core/fwd.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
struct nullspace_data;
struct sparse_ns_factorizer {
    virtual void init(nullspace_data *) {}
    virtual void update(nullspace_data *) {}
};
def_ptr(sparse_ns_factorizer);
} // namespace ns_riccati
} // namespace solver
} // namespace moto

#endif