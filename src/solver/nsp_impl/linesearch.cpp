#include <moto/ocp/impl/sym_data.hpp>
#include <moto/solver/linesearch_config.hpp>
#include <moto/core/workspace_data.hpp>
#define MOTO_NS_RICCATI_IMPL
#include <moto/solver/ns_riccati/generic_solver.hpp>
namespace moto {
namespace solver {
namespace ns_riccati {
void generic_solver::apply_affine_step(ns_riccati_data *cur, workspace_data *_cfg) {
    auto &cfg = _cfg->as<solver::linesearch_config>();
    auto &d = *cur;
    for (auto f : primal_fields) {
        // cur->sym_->value_[f].noalias() += cfg.alpha_primal * d.prim_step[f];
        cur->sym_->integrate(f, d.prim_step[f], cfg.alpha_primal);
        // d.prim_step[f] *= cfg.alpha_primal;
    }
    for (auto f : hard_constr_fields) {
        // cur->dense_->dual_[f].noalias() += cfg.alpha_dual * d.dual_step[f];
        cur->dense_->dual_[f].noalias() += cfg.alpha_primal * d.dual_step[f];
        // d.dual_step[f] *= cfg.alpha_dual;
    }
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto