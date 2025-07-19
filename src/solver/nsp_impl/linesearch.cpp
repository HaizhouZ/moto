#include <moto/solver/linesearch_config.hpp>
#include <moto/solver/ns_riccati/ns_riccati_solve.hpp>

namespace moto {
namespace solver {
namespace ns_riccati {
void line_search_step(ns_node_data *cur, workspace_data *_cfg) {
    auto &cfg = _cfg->get<solver::linesearch_config>();
    auto &d = *cur;
    for (auto f : primal_fields) {
        cur->sym_->value_[f].noalias() += cfg.alpha_primal * d.prim_step[f];
    }
    for (auto f : hard_constr_fields) {
        cur->dense_->dual_[f].noalias() += cfg.alpha_dual * d.dual_step[f];
    }
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto