#include <moto/solver/ns_riccati_solve.hpp>
#include <moto/solver/ineq_soft_solve.hpp>

namespace moto {
namespace nullsp_kkt_solve {
void line_search_step(riccati_data *cur, scalar_t alpha) {
    auto &d = *cur;
    for (auto f : solver::primal_fields) {
        cur->sym_->value_[f].noalias() += alpha * d.prim_step[f];
    }
    for (auto f : solver::hard_constr_fields) {
        cur->dense_->dual_[f].noalias() += alpha * d.dual_step[f];
    }
    ineq_soft_solve::line_search_step(cur, alpha);
}

} // namespace nullsp_kkt_solve
} // namespace moto