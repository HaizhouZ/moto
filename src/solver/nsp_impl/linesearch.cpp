#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>
#include <moto/solver/ineq_soft_solve.hpp>

namespace moto {
namespace ns_riccati {
void line_search_step(ns_node_data *cur, solver::line_search_cfg *cfg) {
    auto &d = *cur;
    for (auto f : primal_fields) {
        cur->sym_->value_[f].noalias() += cfg->alpha_primal * d.prim_step[f];
    }
    for (auto f : hard_constr_fields) {
        cur->dense_->dual_[f].noalias() += cfg->alpha_dual * d.dual_step[f];
    }
    ineq_soft_solve::line_search_step(cur, *cfg);
}

} // namespace ns_riccati
} // namespace moto