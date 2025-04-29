#include <atri/solver/ns_sqp.hpp>

namespace atri {
void nullspace_riccati_solver::post_solving_steps() {
    // compute zero order sensitivities and the rest 1st order ones
#pragma omp parallel
    for (int i = 0; i < nodes_.size(); i++) {
        // collect constraint residuals and jacobians
        auto cur = nodes_[i];
        auto &d = get_data(cur);
        auto &_approx = d.raw_data_.approx_;
        
    }
}
} // namespace atri