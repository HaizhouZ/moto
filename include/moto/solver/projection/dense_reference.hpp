#ifndef MOTO_SOLVER_PROJECTION_DENSE_REFERENCE_HPP
#define MOTO_SOLVER_PROJECTION_DENSE_REFERENCE_HPP

#include <moto/solver/projection/types.hpp>

namespace moto::solver::ns_riccati {
struct ns_riccati_data;
enum rank_status : int;
}

namespace moto::solver::projection {

struct dense_snapshot {
    assembled_problem prob;
    stage_outputs out;

    void clear_numeric() {
        prob.clear_numeric();
        out.clear_numeric();
    }
};

void snapshot_dense_factorization(const ns_riccati::ns_riccati_data &data,
                                  dense_snapshot &snapshot);
void snapshot_dense_reduced_step(const ns_riccati::ns_riccati_data &data,
                                 dense_snapshot &snapshot);
void snapshot_dense_duals(const ns_riccati::ns_riccati_data &data,
                          dense_snapshot &snapshot);

} // namespace moto::solver::projection

#endif
