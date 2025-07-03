#include <moto/solver/ipm.hpp>
#include <moto/solver/ipm_constr.hpp>

namespace moto {
namespace ipm {
void post_rollout(solver::solver_data *cur) {
    // get all ipm constraints
    for (auto &p : cur->sparse_[__ineq_x]) {
        auto &d = static_cast<ipm_data &>(*p);
        // update slack newton step
        size_t arg_idx = 0;
        // compute linear step
        for (const auto &arg : d.f_.in_args()) {
            if (arg->field_ < field::num_prim) {
                d.d_slack_.noalias() -= d.jac_[arg_idx] * cur->ocp_->extract(cur->prim_rollout_[arg_idx], *arg);
            }
            arg_idx++;
        }
        d.d_slack_.noalias() -= d.v_ + d.slack_; // +r_g
        // update dual newton step
        d.d_multipler_.array() = -d.multiplier_.cwiseProduct(d.slack_).array() - d.mu_ - d.diag_scaling.array() * d.d_slack_.array();
    }
}

} // namespace ipm
} // namespace moto
