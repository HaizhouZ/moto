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
        // cur->sym_->value_[f].noalias() += cfg.alpha_primal * d.trial_prim_step[f];
        cur->sym_->integrate(f, d.trial_prim_step[f], cfg.alpha_primal);
        // d.trial_prim_step[f] *= cfg.alpha_primal;
    }
    const scalar_t dual_alpha = cfg.dual_alpha_for_eq();
    for (auto f : hard_constr_fields) {
        cur->dense_->dual_[f].noalias() += dual_alpha * d.trial_dual_step[f];
    }
}

} // namespace ns_riccati
} // namespace solver
} // namespace moto